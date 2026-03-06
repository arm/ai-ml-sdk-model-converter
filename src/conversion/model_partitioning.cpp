/*
 * SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "include/passes.hpp"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "vgf-dialect/VGFDialect.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Error.h"

#include <nlohmann/json.hpp>

#include <cstring>
#include <optional>
#include <queue>
#include <vector>

namespace mlir {
namespace model_converter_passes {
#define GEN_PASS_DEF_MODELPARTITIONINGPASS
#include "passes.hpp.inc"
namespace {
using json = nlohmann::json;

struct FuncOpRewriter : public OpConversionPattern<func::FuncOp> {
    using OpConversionPattern<func::FuncOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto sequenceOp =
            vgf::SequenceOp::create(rewriter, funcOp.getLoc(), adaptor.getSymName(), adaptor.getFunctionType(),
                                    adaptor.getArgAttrsAttr(), adaptor.getResAttrsAttr());
        sequenceOp->setAttrs(adaptor.getAttributes());
        rewriter.inlineRegionBefore(funcOp.getBody(), sequenceOp.getBody(), sequenceOp.end());
        rewriter.eraseOp(funcOp);
        return success();
    }
};

struct ReturnOpRewriter : public OpConversionPattern<func::ReturnOp> {
    using OpConversionPattern<func::ReturnOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(func::ReturnOp returnOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<vgf::SequenceOutputOp>(returnOp, adaptor.getOperands());
        return success();
    }
};

struct TosaCustomOpRewriter : public OpConversionPattern<tosa::CustomOp> {
    using OpConversionPattern<tosa::CustomOp>::OpConversionPattern;

  public:
    explicit TosaCustomOpRewriter(MLIRContext *context, bool analysis = false, PatternBenefit benefit = 1)
        : OpConversionPattern<tosa::CustomOp>(context, benefit), analysis(analysis) {}

    LogicalResult matchAndRewrite(tosa::CustomOp customOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        json map;
        try {
            map = json::parse(adaptor.getImplementationAttrs().str());
        } catch (...) {
            llvm::errs() << "Invalid JSON implementation_attrs in tosa.custom op at " << customOp->getLoc() << "\n";
            return failure();
        }
        if (!map.is_object()) {
            llvm::errs() << "implementation_attrs must be a JSON object in tosa.custom op at " << customOp->getLoc()
                         << "\n";
            return failure();
        }

        StringAttr shaderNameAttr;
        StringAttr entryPointAttr;
        DenseI64ArrayAttr inputBindingsAttr;
        DenseI64ArrayAttr outputBindingsAttr;
        DenseI64ArrayAttr inputDescriptorSetsAttr;
        DenseI64ArrayAttr outputDescriptorSetsAttr;
        ArrayAttr inputVkDescriptorTypesAttr;
        ArrayAttr outputVkDescriptorTypesAttr;
        ArrayAttr inputVkFormatsAttr;
        ArrayAttr outputVkFormatsAttr;
        DenseI64ArrayAttr workgroupSizesAttr;
        std::optional<StringAttr> shaderLanguageAttr;
        std::optional<Attribute> shaderCodeAttr;

        shaderNameAttr = rewriter.getStringAttr(adaptor.getDomainName().str() + "::" + adaptor.getOperatorName().str());

        if (!fetch(map, "entry_point", [&](const json &reference) {
                if (!reference.is_string()) {
                    return false;
                }
                entryPointAttr = rewriter.getStringAttr(reference.get<std::string>());
                return true;
            })) {
            llvm::errs() << "Missing attribute or invalid value for entry_point in tosa.custom op at "
                         << customOp->getLoc() << "\n";
            return failure();
        }

        if (!parseIO(rewriter, map, "input", customOp->getNumOperands(), inputBindingsAttr, inputDescriptorSetsAttr,
                     inputVkDescriptorTypesAttr, inputVkFormatsAttr)) {
            llvm::errs() << "Missing input attribute(s) or invalid value in tosa.custom op at " << customOp->getLoc()
                         << "\n";
            return failure();
        }

        if (!parseIO(rewriter, map, "output", customOp->getNumResults(), outputBindingsAttr, outputDescriptorSetsAttr,
                     outputVkDescriptorTypesAttr, outputVkFormatsAttr)) {
            llvm::errs() << "Missing output attribute(s) or invalid value in tosa.custom op at " << customOp->getLoc()
                         << "\n";
            return failure();
        }

        if (!fetch(map, "workgroup_sizes", [&](const json &reference) {
                SmallVector<int64_t, 3> workgroupSizes;
                if (!reference.is_array()) {
                    return false;
                }
                if (reference.size() != 3) {
                    return false;
                }
                for (const auto &value : reference) {
                    if (!value.is_number_integer() && !value.is_number_unsigned()) {
                        return false;
                    }
                    workgroupSizes.push_back(value.get<int64_t>());
                }
                workgroupSizesAttr = rewriter.getDenseI64ArrayAttr(workgroupSizes);
                return true;
            })) {
            llvm::errs() << "Missing attribute or invalid value for workgroup_sizes in tosa.custom op at "
                         << customOp->getLoc() << "\n";
            return failure();
        }

        if (has(map, "shader_language")) {
            if (!fetch(map, "shader_language", [&](const json &reference) {
                    if (!reference.is_string()) {
                        return false;
                    }
                    shaderLanguageAttr = rewriter.getStringAttr(reference.get<std::string>());
                    return true;
                })) {
                llvm::errs() << "Invalid value for shader_language attribute in tosa.custom op at "
                             << customOp->getLoc() << "\n";
                return failure();
            }
        }

        if (has(map, "shader_code")) {
            if (!fetch(map, "shader_code", [&](const json &reference) {
                    if (!reference.is_string()) {
                        return false;
                    }
                    const auto code = reference.get<std::string>();
                    if (shaderLanguageAttr.has_value() &&
                        (shaderLanguageAttr.value().str() == "GLSL" || shaderLanguageAttr.value().str() == "HLSL")) {
                        shaderCodeAttr = rewriter.getStringAttr(code);
                        return true;
                    }

                    if (!shaderLanguageAttr.has_value() || shaderLanguageAttr.value().str() != "SPIR-V") {
                        return false;
                    }

                    std::vector<char> binaryCode;
                    llvm::Error decodeError = llvm::decodeBase64(code, binaryCode);
                    if (decodeError) {
                        llvm::consumeError(std::move(decodeError));
                        return false;
                    }
                    if (binaryCode.size() % sizeof(int32_t) != 0) {
                        return false;
                    }
                    ArrayRef<int32_t> array(reinterpret_cast<int32_t *>(binaryCode.data()),
                                            binaryCode.size() / sizeof(int32_t));
                    shaderCodeAttr = rewriter.getDenseI32ArrayAttr(array);
                    return true;
                })) {

                llvm::errs() << "Invalid value for shader_code attribute in tosa.custom op at " << customOp->getLoc()
                             << "\n";
                return failure();
            }
        }

        if (shaderLanguageAttr.has_value() != shaderCodeAttr.has_value()) {
            llvm::errs() << "shader_language and shader_code must both be set or both be unset in tosa.custom op at "
                         << customOp->getLoc() << "\n";
            return failure();
        }

        rewriter.replaceOpWithNewOp<vgf::ShaderPlaceholderOp>(
            customOp, customOp.getResultTypes(), shaderNameAttr, entryPointAttr, inputBindingsAttr, outputBindingsAttr,
            inputDescriptorSetsAttr, outputDescriptorSetsAttr, inputVkDescriptorTypesAttr, outputVkDescriptorTypesAttr,
            inputVkFormatsAttr, outputVkFormatsAttr, workgroupSizesAttr, shaderLanguageAttr.value_or(nullptr),
            shaderCodeAttr.value_or(nullptr), adaptor.getOperands());

        if (analysis) {
            llvm::errs() << "Successfully lowered: " << customOp->getName() << " at " << customOp->getLoc() << "\n";
        }
        return success();
    }

  private:
    bool analysis;

    const json *get(const json &map, const std::string &key) const {
        const auto it = map.find(key);
        if (it == map.end() || it->is_null()) {
            return nullptr;
        }
        return &*it;
    }

    bool has(const json &map, const std::string &key) const { return get(map, key) != nullptr; }

    template <typename Fn> bool fetch(const json &map, const std::string &key, Fn &&callback) const {
        const auto *value = get(map, key);
        return value != nullptr && callback(*value);
    }

    bool parseIO(ConversionPatternRewriter &rewriter, const json &map, const std::string &prefix, const unsigned numIOs,
                 DenseI64ArrayAttr &bindingsAttr, DenseI64ArrayAttr &descriptorSetsAttr,
                 ArrayAttr &vkDescriptorTypesAttr, ArrayAttr &vkFormatsAttr) const {
        SmallVector<int64_t, 8> bindings;
        SmallVector<int64_t, 8> descriptorSets;
        SmallVector<Attribute, 8> vkDescriptorTypes;
        SmallVector<Attribute, 8> vkFormats;

        for (unsigned i = 0; i < numIOs; ++i) {
            const auto index = std::to_string(i);
            std::string ioKeyPrefix = prefix;
            ioKeyPrefix += "_";
            ioKeyPrefix += index;

            std::string key = ioKeyPrefix;
            key += "_binding";
            if (!fetch(map, key, [&](const json &reference) {
                    if (!reference.is_number_integer() && !reference.is_number_unsigned()) {
                        return false;
                    }
                    bindings.push_back(reference.get<int64_t>());
                    return true;
                })) {
                return false;
            }

            key = ioKeyPrefix;
            key += "_descriptorset";
            if (!fetch(map, key, [&](const json &reference) {
                    if (!reference.is_number_integer() && !reference.is_number_unsigned()) {
                        return false;
                    }
                    descriptorSets.push_back(reference.get<int64_t>());
                    return true;
                })) {
                return false;
            }

            key = ioKeyPrefix;
            key += "_vkdescriptortype";
            if (!fetch(map, key, [&](const json &reference) {
                    if (!reference.is_string()) {
                        return false;
                    }
                    vkDescriptorTypes.push_back(rewriter.getStringAttr(reference.get_ref<const std::string &>()));
                    return true;
                })) {
                return false;
            }

            key = ioKeyPrefix;
            key += "_vkformat";
            if (!fetch(map, key, [&](const json &reference) {
                    if (!reference.is_string()) {
                        return false;
                    }
                    vkFormats.push_back(rewriter.getStringAttr(reference.get_ref<const std::string &>()));
                    return true;
                })) {
                return false;
            }
        }

        bindingsAttr = rewriter.getDenseI64ArrayAttr(bindings);
        descriptorSetsAttr = rewriter.getDenseI64ArrayAttr(descriptorSets);
        vkDescriptorTypesAttr = rewriter.getArrayAttr(vkDescriptorTypes);
        vkFormatsAttr = rewriter.getArrayAttr(vkFormats);
        return true;
    }
};

void insertPartitionOpInMap(const int64_t id, Operation *op, DenseMap<int64_t, SmallVector<Operation *>> &map) {
    if (!map.contains(id)) {
        map[id] = SmallVector<Operation *>();
    }
    map[id].push_back(op);
}

bool comparePartitionResultIndex(const Value &a, const Value &b) {
    int64_t a_idx = 0;
    int64_t b_idx = 0;

    if (auto attr = a.getDefiningOp()->getAttrOfType<IntegerAttr>("graph_partition_sequence_output_index")) {
        a_idx = attr.getInt();
    }
    if (auto attr = b.getDefiningOp()->getAttrOfType<IntegerAttr>("graph_partition_sequence_output_index")) {
        b_idx = attr.getInt();
    }

    return a_idx < b_idx;
}

void insertPartitionResultInMap(const int64_t id, Value value, DenseMap<int64_t, SmallVector<Value>> &map) {
    if (!map.contains(id)) {
        map[id] = SmallVector<Value>();
    }
    auto *position = std::lower_bound(map[id].begin(), map[id].end(), value, comparePartitionResultIndex);
    map[id].insert(position, value);
}

SmallVector<Value> collectInputs(const SmallVector<Operation *> &ops) {
    DenseSet<Operation *> knownOps(ops.begin(), ops.end());
    DenseSet<Value> seenInputs;
    SmallVector<Value> inputs;
    for (Operation *op : ops) {
        for (auto operand : op->getOperands()) {
            Operation *defOp = operand.getDefiningOp();
            if (!knownOps.contains(defOp) && seenInputs.insert(operand).second) {
                inputs.push_back(operand);
            }
        }
    }

    return inputs;
}

void deleteOldOps(ModuleOp moduleOp) {
    std::vector<Operation *> opsToDelete;
    moduleOp.walk([&](Operation *op) {
        if (auto attr = op->getAttrOfType<BoolAttr>("delete")) {
            if (attr.getValue()) {
                opsToDelete.push_back(op);
            }
        }
    });
    for (auto it = opsToDelete.rbegin(); it != opsToDelete.rend(); ++it) {
        (*it)->erase();
    }
}

class ModelPartitioningPass : public impl::ModelPartitioningPassBase<ModelPartitioningPass> {
  public:
    using impl::ModelPartitioningPassBase<ModelPartitioningPass>::ModelPartitioningPassBase;

    void runOnOperation() override {
        mlir::ModuleOp moduleOp = getOperation();
        MLIRContext *context = &getContext();
        int64_t highestPartitionId = -1;

        DenseMap<int64_t, SmallVector<Operation *>> partitionIdToOp;
        DenseMap<int64_t, SmallVector<Value>> partitionIdToResults;
        moduleOp.walk([&](Operation *op) {
            if (llvm::isa<mlir::ModuleOp>(op) || llvm::isa<mlir::func::FuncOp>(op) ||
                llvm::isa<mlir::func::ReturnOp>(op)) {
                return;
            }
            auto partitionAttr = op->getAttrOfType<IntegerAttr>("graph_partition_id");
            int64_t partitionId = partitionAttr.getInt();
            insertPartitionOpInMap(partitionId, op, partitionIdToOp);
            highestPartitionId = std::max(highestPartitionId, partitionId);

            auto leafAttr = op->getAttrOfType<BoolAttr>("graph_partition_leaf_node");
            if (leafAttr.getValue()) {
                for (Value value : op->getResults()) {
                    insertPartitionResultInMap(partitionId, value, partitionIdToResults);
                }
            }
        });

        moduleOp.walk([&](func::FuncOp funcOp) {
            OpBuilder builder(context);
            const Type tUI32 = IntegerType::get(context, 32, IntegerType::SignednessSemantics::Unsigned);
            Operation *oldTerminator = funcOp.getBody().front().getTerminator();
            builder.setInsertionPoint(oldTerminator);

            IRMapping externalMapping;
            for (int64_t partitionId = 0; partitionId <= highestPartitionId; ++partitionId) {
                SmallVector<Operation *> partitionOps = partitionIdToOp[partitionId];

                SmallVector<Value> inputs;
                // This ensures that unused arguments are passed through when there is one segment(partition)
                if (highestPartitionId == 0) {
                    auto args = funcOp.getArguments();
                    std::copy(args.begin(), args.end(), std::back_inserter(inputs));
                } else {
                    inputs = collectInputs(partitionOps);
                }
                SmallVector<Value> results = partitionIdToResults[partitionId];

                const std::string segmentName = "graph_partition_" + std::to_string(partitionId);
                const FunctionType segmentFunctionType =
                    builder.getFunctionType(ValueRange(inputs).getTypes(), ValueRange(results).getTypes());

                bool isComputeSegment = partitionOps.size() == 1 && llvm::isa<tosa::CustomOp>(partitionOps[0]);
                const auto segmentType = isComputeSegment ? vgf::SegmentTypeEnum::COMPUTE : vgf::SegmentTypeEnum::GRAPH;

                auto segmentOp = vgf::SegmentOp::create(builder, funcOp.getLoc(), segmentName, segmentType,
                                                        segmentFunctionType, nullptr, nullptr);
                segmentOp->setAttr("segment_id", IntegerAttr::get(tUI32, partitionId));
                {
                    OpBuilder::InsertionGuard segmentGuard{builder};
                    Block *segmentBlock = segmentOp.addEntryBlock();
                    builder.setInsertionPoint(segmentBlock, segmentBlock->end());

                    if (isComputeSegment) {
                        IRMapping segmentMapping;
                        for (auto [input, segmentOpArg] : llvm::zip(inputs, segmentOp.getArguments())) {
                            segmentMapping.map(input, segmentOpArg);
                        }

                        builder.clone(*partitionOps[0], segmentMapping);
                        partitionOps[0]->setAttr("delete", BoolAttr::get(context, true));

                        llvm::SmallVector<Value, 4> segmentResults;
                        std::transform(results.begin(), results.end(), std::back_inserter(segmentResults),
                                       [&](Value value) { return segmentMapping.lookupOrDefault(value); });
                        vgf::SegmentOutputOp::create(builder, segmentOp.getLoc(), segmentResults);
                    } else {
                        auto newFuncOp =
                            func::FuncOp::create(builder, funcOp.getLoc(), segmentName, segmentFunctionType);
                        newFuncOp->setAttr("segment_id", IntegerAttr::get(tUI32, partitionId));

                        llvm::SmallVector<Value, 0> segmentResults;
                        vgf::SegmentOutputOp::create(builder, segmentOp.getLoc(), segmentResults);

                        {
                            OpBuilder::InsertionGuard funcGuard{builder};
                            Block *funcBlock = newFuncOp.addEntryBlock();
                            builder.setInsertionPoint(funcBlock, funcBlock->end());
                            IRMapping funcMapping;
                            for (auto [input, newFunOpArg] : llvm::zip(inputs, newFuncOp.getArguments())) {
                                funcMapping.map(input, newFunOpArg);
                            }

                            for (Operation *op : partitionOps) {
                                builder.clone(*op, funcMapping);
                                op->setAttr("delete", BoolAttr::get(context, true));
                            }

                            llvm::SmallVector<Value, 4> funcResults;
                            std::transform(results.begin(), results.end(), std::back_inserter(funcResults),
                                           [&](Value value) { return funcMapping.lookupOrDefault(value); });
                            func::ReturnOp::create(builder, newFuncOp.getLoc(), funcResults);
                        }
                    }
                }

                llvm::SmallVector<Value, 4> runInputs;
                std::transform(inputs.begin(), inputs.end(), std::back_inserter(runInputs),
                               [&](Value value) { return externalMapping.lookupOrDefault(value); });
                auto segmentRunOp = vgf::SegmentRunOp::create(builder, segmentOp.getLoc(), segmentOp.getResultTypes(),
                                                              SymbolRefAttr::get(segmentOp), ValueRange(runInputs));
                segmentRunOp->setAttr("segment_id", IntegerAttr::get(tUI32, partitionId));

                for (auto [result, segmentRunOpArg] : llvm::zip(results, segmentRunOp.getResults())) {
                    externalMapping.map(result, segmentRunOpArg);
                }
            }

            builder.clone(*oldTerminator, externalMapping);
            oldTerminator->setAttr("delete", BoolAttr::get(context, true));
        });

        deleteOldOps(moduleOp);

        ConversionTarget target(*context);
        target.addDynamicallyLegalOp<func::FuncOp>(
            [](func::FuncOp op) { return !llvm::isa<ModuleOp>(op->getParentOp()); });
        target.addDynamicallyLegalOp<func::ReturnOp>(
            [](func::ReturnOp op) { return !llvm::isa<ModuleOp>(op->getParentOp()->getParentOp()); });
        target.addIllegalOp<tosa::CustomOp>();
        target.addLegalDialect<vgf::VGFDialect, func::FuncDialect>();
        RewritePatternSet patterns(context);
        patterns.add<FuncOpRewriter, ReturnOpRewriter>(context);
        patterns.add<TosaCustomOpRewriter>(context, analysis);
        if (applyPartialConversion(moduleOp, target, std::move(patterns)).failed()) {
            return signalPassFailure();
        }
    }
};

} // namespace

} // namespace model_converter_passes
} // namespace mlir
