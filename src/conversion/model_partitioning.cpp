/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "include/passes.hpp"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "vgf-dialect/VGFDialect.h"

#include <flatbuffers/flexbuffers.h>

#include <optional>
#include <queue>

namespace mlir {
namespace model_converter_passes {
namespace {

std::string chop(std::string &text, const std::string &c) {
    size_t pos = text.find_first_of(c);
    if (std::string::npos == pos) {
        return "";
    }
    std::string pre = text.substr(0, pos);
    text = text.substr(pos + 1);
    return pre;
}

LogicalResult parseInts(SmallVector<int64_t, 3> &output, std::string &&text, const std::string &sep) {
    try {
        std::string token = chop(text, sep);
        while (!token.empty()) {
            output.push_back(std::stol(token));
            token = chop(text, sep);
        }
        output.push_back(std::stol(text));
        return success();
    } catch (...) {
        return failure();
    }
}

struct FuncOpRewriter : public OpConversionPattern<func::FuncOp> {
    using OpConversionPattern<func::FuncOp>::OpConversionPattern;
    LogicalResult matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto sequenceOp =
            rewriter.create<vgf::SequenceOp>(funcOp.getLoc(), adaptor.getSymName(), adaptor.getFunctionType(),
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
        auto implementationAttrs = adaptor.getImplementationAttrs();
        auto root = flexbuffers::GetRoot(reinterpret_cast<const uint8_t *>(implementationAttrs.data()),
                                         implementationAttrs.size());
        if (!root.IsMap()) {
            return failure();
        }
        auto map = root.AsMap();

        StringAttr shaderNameAttr;
        StringAttr entryPointAttr;
        DenseI64ArrayAttr inputBindingsAttr;
        DenseI64ArrayAttr outputBindingsAttr;
        IntegerAttr inputDescriptorSetAttr;
        IntegerAttr outputDescriptorSetAttr;
        ArrayAttr inputVkDescriptorTypesAttr;
        ArrayAttr outputVkDescriptorTypesAttr;
        ArrayAttr inputVkFormatsAttr;
        ArrayAttr outputVkFormatsAttr;
        DenseI64ArrayAttr workgroupSizesAttr;
        std::optional<DenseI32ArrayAttr> shaderCodeAttr;

        shaderNameAttr = rewriter.getStringAttr(adaptor.getDomainName().str() + "::" + adaptor.getOperatorName().str());

        if (failed(fetch(map, "entry_point", [&](auto &reference) {
                if (!reference.IsString()) {
                    return failure();
                }
                entryPointAttr = rewriter.getStringAttr(reference.AsString().str());
                return success();
            }))) {
            llvm::errs() << "Missing attribute or invalid value for entry_point in tosa.custom op at "
                         << customOp->getLoc() << "\n";
            return failure();
        }

        if (failed(parseIO(rewriter, map, "input", customOp->getNumOperands(), inputBindingsAttr,
                           inputDescriptorSetAttr, inputVkDescriptorTypesAttr, inputVkFormatsAttr))) {
            llvm::errs() << "Missing input attribute(s) or invalid value in tosa.custom op at " << customOp->getLoc()
                         << "\n";
            return failure();
        }

        if (failed(parseIO(rewriter, map, "output", customOp->getNumResults(), outputBindingsAttr,
                           outputDescriptorSetAttr, outputVkDescriptorTypesAttr, outputVkFormatsAttr,
                           /* offset=*/customOp->getNumOperands()))) {
            llvm::errs() << "Missing output attribute(s) or invalid value in tosa.custom op at " << customOp->getLoc()
                         << "\n";
            return failure();
        }

        if (failed(fetch(map, "workgroup_sizes", [&](auto &reference) {
                if (!reference.IsString()) {
                    return failure();
                }
                SmallVector<int64_t, 3> workgroupSizes;
                if (failed(parseInts(workgroupSizes, reference.AsString().str(), ","))) {
                    return failure();
                }
                workgroupSizesAttr = rewriter.getDenseI64ArrayAttr(workgroupSizes);
                return success();
            }))) {
            llvm::errs() << "Missing attribute or invalid value for workgroup_sizes in tosa.custom op at "
                         << customOp->getLoc() << "\n";
            return failure();
        }

        if (failed(fetch(
                map, "shader_code",
                [&](auto &reference) {
                    if (!reference.IsString()) {
                        return failure();
                    }
                    // FIXME: here we should either decode the base64 string or get the binary from the metadata
                    SmallVector<int8_t> code /* = ??? */;
                    if (code.size() % sizeof(int32_t) != 0) {
                        return failure();
                    }
                    ArrayRef<int32_t> spv(reinterpret_cast<const int32_t *>(code.data()),
                                          code.size() / sizeof(int32_t));
                    shaderCodeAttr = rewriter.getDenseI32ArrayAttr(spv);
                    return success();
                },
                true))) {

            llvm::errs() << "Invalid value for shader_code attribute in tosa.custom op at " << customOp->getLoc()
                         << "\n";
            return failure();
        }

        rewriter.replaceOpWithNewOp<vgf::ShaderPlaceholderOp>(
            customOp, customOp.getResultTypes(), shaderNameAttr, entryPointAttr, inputBindingsAttr, outputBindingsAttr,
            inputDescriptorSetAttr, outputDescriptorSetAttr, inputVkDescriptorTypesAttr, outputVkDescriptorTypesAttr,
            inputVkFormatsAttr, outputVkFormatsAttr, workgroupSizesAttr, shaderCodeAttr.value_or(nullptr),
            adaptor.getOperands());

        if (analysis) {
            llvm::errs() << "Successfully lowered: " << customOp->getName() << " at " << customOp->getLoc() << "\n";
        }
        return success();
    }

  private:
    bool analysis;

    LogicalResult fetch(const flexbuffers::Map &map, const std::string &key,
                        std::function<LogicalResult(flexbuffers::Reference &)> callback, bool optional = false) const {
        auto reference = map[key];
        if (reference.IsNull()) {
            return optional ? success() : failure();
        }
        if (failed(callback(reference))) {
            return failure();
        }
        return success();
    }

    LogicalResult parseIO(ConversionPatternRewriter &rewriter, const flexbuffers::Map &map, const std::string &prefix,
                          const unsigned numIOs, DenseI64ArrayAttr &bindingsAttr, IntegerAttr &descriptorSetAttr,
                          ArrayAttr &vkDescriptorTypesAttr, ArrayAttr &vkFormatsAttr, const unsigned offset = 0) const {
        SmallVector<int64_t, 8> bindings;
        SmallVector<StringRef, 8> vkDescriptorTypes;
        SmallVector<StringRef, 8> vkFormats;
        int64_t descriptorSet = 0; // FIXME: no default should be needed since metadata should be mandatory

        const auto descriptorSetKey = prefix + "_descriptor_set";
        if (failed(fetch(
                map, descriptorSetKey,
                [&](auto &reference) {
                    if (!reference.IsIntOrUint()) {
                        return failure();
                    }
                    descriptorSet = reference.AsInt64();
                    return success();
                },
                /* FIXME: this should be mandatory */
                true))) {
            return failure();
        }

        for (unsigned i = 0; i < numIOs; ++i) {
            const auto bindingKey = prefix + "<" + std::to_string(i) + ">" + "_binding";
            if (failed(fetch(
                    map, bindingKey,
                    [&](auto &reference) {
                        if (!reference.IsIntOrUint()) {
                            return failure();
                        }
                        bindings.push_back(reference.AsInt64());
                        return success();
                    },
                    /* FIXME: this should be mandatory */
                    true))) {
                return failure();
            }

            const auto vkDescriptorTypeKey = prefix + "<" + std::to_string(i) + ">" + "_vkdescriptortype";
            if (failed(fetch(map, vkDescriptorTypeKey, [&](auto &reference) {
                    if (!reference.IsString()) {
                        return failure();
                    }
                    auto str = reference.AsString();
                    vkDescriptorTypes.push_back(StringRef(str.c_str(), str.length()));
                    return success();
                }))) {
                return failure();
            }
            const auto vkFormatKey = prefix + "<" + std::to_string(i) + ">" + "_vkformat";
            if (failed(fetch(map, vkFormatKey, [&](auto &reference) {
                    if (!reference.IsString()) {
                        return failure();
                    }
                    auto str = reference.AsString();
                    vkFormats.push_back(StringRef(str.c_str(), str.length()));
                    return success();
                }))) {
                return failure();
            }
        }

        // FIXME: this should not be needed since the binding_id should be mandatory metadata
        if (bindings.size() < numIOs) {
            bindings.clear();
            for (unsigned i = 0; i < numIOs; ++i) {
                bindings.push_back(i + offset);
            }
        }

        bindingsAttr = rewriter.getDenseI64ArrayAttr(bindings);
        descriptorSetAttr = rewriter.getIntegerAttr(rewriter.getI64Type(), descriptorSet);
        vkDescriptorTypesAttr = rewriter.getStrArrayAttr(vkDescriptorTypes);
        vkFormatsAttr = rewriter.getStrArrayAttr(vkFormats);
        return success();
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

    if (auto attr = a.getDefiningOp()->getAttrOfType<IntegerAttr>("graph_partition_sequence_output_index"))
        a_idx = attr.getInt();
    if (auto attr = b.getDefiningOp()->getAttrOfType<IntegerAttr>("graph_partition_sequence_output_index"))
        b_idx = attr.getInt();

    return a_idx < b_idx;
}

void insertPartitionResultInMap(const int64_t id, Value value, DenseMap<int64_t, SmallVector<Value>> &map) {
    if (!map.contains(id)) {
        map[id] = SmallVector<Value>();
    }
    auto position = std::lower_bound(map[id].begin(), map[id].end(), value, comparePartitionResultIndex);
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
        if (BoolAttr attr = op->getAttrOfType<BoolAttr>("delete")) {
            if (attr.getValue()) {
                opsToDelete.push_back(op);
            }
        }
    });
    for (auto it = opsToDelete.rbegin(); it != opsToDelete.rend(); ++it) {
        (*it)->erase();
    }
}

class ModelPartitioningPass : public ModelPartitioningPassBase<ModelPartitioningPass> {
  public:
    explicit ModelPartitioningPass(const ModelPartitioningPassOptions &options) : analysis(options.analysis) {}

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

                auto segmentOp = builder.create<vgf::SegmentOp>(funcOp.getLoc(), segmentName, segmentType,
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
                        builder.create<vgf::SegmentOutputOp>(segmentOp.getLoc(), segmentResults);
                    } else {
                        auto newFuncOp =
                            builder.create<func::FuncOp>(funcOp.getLoc(), segmentName, segmentFunctionType);
                        newFuncOp->setAttr("segment_id", IntegerAttr::get(tUI32, partitionId));

                        llvm::SmallVector<Value, 0> segmentResults;
                        builder.create<vgf::SegmentOutputOp>(segmentOp.getLoc(), segmentResults);

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
                            builder.create<func::ReturnOp>(newFuncOp.getLoc(), funcResults);
                        }
                    }
                }

                llvm::SmallVector<Value, 4> runInputs;
                std::transform(inputs.begin(), inputs.end(), std::back_inserter(runInputs),
                               [&](Value value) { return externalMapping.lookupOrDefault(value); });
                auto segmentRunOp =
                    builder.create<vgf::SegmentRunOp>(segmentOp.getLoc(), segmentOp.getResultTypes(),
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

  private:
    bool analysis;
};

} // namespace

std::unique_ptr<Pass> createModelPartitioningPass(ModelPartitioningPassOptions options) {
    return std::make_unique<ModelPartitioningPass>(options);
}

void registerModelPartitioningPass() {
    PassRegistration<ModelPartitioningPass>(
        []() -> std::unique_ptr<Pass> { return createModelPartitioningPass({false}); });
}

} // namespace model_converter_passes
} // namespace mlir
