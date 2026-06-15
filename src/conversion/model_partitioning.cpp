/*
 * SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "include/custom_op_domains.hpp"
#include "include/passes.hpp"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "model_partition_attrs.hpp"
#include "vgf-dialect/VGFDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Error.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <optional>
#include <utility>
#include <vector>

namespace mlir::model_converter_passes {
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
        if (!isVulkanCustomShaderOp(customOp)) {
            return failure();
        }

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
        ArrayAttr inputSamplerConfigsAttr;
        ArrayAttr outputSamplerConfigsAttr;
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
                     inputVkDescriptorTypesAttr, inputVkFormatsAttr, inputSamplerConfigsAttr)) {
            llvm::errs() << "Missing input attribute(s) or invalid value in tosa.custom op at " << customOp->getLoc()
                         << "\n";
            return failure();
        }

        if (!parseIO(rewriter, map, "output", customOp->getNumResults(), outputBindingsAttr, outputDescriptorSetsAttr,
                     outputVkDescriptorTypesAttr, outputVkFormatsAttr, outputSamplerConfigsAttr)) {
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
            inputVkFormatsAttr, outputVkFormatsAttr, inputSamplerConfigsAttr, outputSamplerConfigsAttr,
            workgroupSizesAttr, shaderLanguageAttr.value_or(nullptr), shaderCodeAttr.value_or(nullptr),
            adaptor.getOperands());

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
                 ArrayAttr &vkDescriptorTypesAttr, ArrayAttr &vkFormatsAttr, ArrayAttr &samplerConfigsAttr) const {
        SmallVector<int64_t, 8> bindings;
        SmallVector<int64_t, 8> descriptorSets;
        SmallVector<Attribute, 8> vkDescriptorTypes;
        SmallVector<Attribute, 8> vkFormats;
        SmallVector<Attribute, 8> samplerConfigs;
        bool hasAnySamplerConfig = false;

        auto appendUnsetSamplerConfig = [&]() { samplerConfigs.push_back(rewriter.getDictionaryAttr({})); };

        auto parseRequiredSamplerString = [&](const json &samplerConfig, const char *key, NamedAttrList &output) {
            return fetch(samplerConfig, key, [&](const json &reference) {
                if (!reference.is_string()) {
                    return false;
                }
                output.append(key, rewriter.getStringAttr(reference.get_ref<const std::string &>()));
                return true;
            });
        };

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

            key = ioKeyPrefix;
            key += "_sampler";
            const auto samplerIt = map.find(key);
            if (samplerIt == map.end()) {
                appendUnsetSamplerConfig();
                continue;
            }

            const json &samplerConfig = *samplerIt;
            if (!samplerConfig.is_object()) {
                return false;
            }
            NamedAttrList samplerConfigAttrs;
            if (!parseRequiredSamplerString(samplerConfig, "min_filter", samplerConfigAttrs) ||
                !parseRequiredSamplerString(samplerConfig, "mag_filter", samplerConfigAttrs) ||
                !parseRequiredSamplerString(samplerConfig, "address_mode_u", samplerConfigAttrs) ||
                !parseRequiredSamplerString(samplerConfig, "address_mode_v", samplerConfigAttrs) ||
                !parseRequiredSamplerString(samplerConfig, "border_color", samplerConfigAttrs)) {
                return false;
            }
            samplerConfigs.push_back(samplerConfigAttrs.getDictionary(rewriter.getContext()));
            hasAnySamplerConfig = true;
        }

        bindingsAttr = rewriter.getDenseI64ArrayAttr(bindings);
        descriptorSetsAttr = rewriter.getDenseI64ArrayAttr(descriptorSets);
        vkDescriptorTypesAttr = rewriter.getArrayAttr(vkDescriptorTypes);
        vkFormatsAttr = rewriter.getArrayAttr(vkFormats);
        samplerConfigsAttr = hasAnySamplerConfig ? rewriter.getArrayAttr(samplerConfigs) : nullptr;
        return true;
    }
};

int64_t getSequenceOutputIndex(Value value) {
    auto result = llvm::dyn_cast<OpResult>(value);
    if (!result) {
        return -1;
    }

    auto attr = result.getDefiningOp()->getAttrOfType<DenseI64ArrayAttr>(graphPartitionSequenceOutputIndicesAttrName);
    if (!attr || result.getResultNumber() >= attr.size()) {
        return -1;
    }

    return attr[result.getResultNumber()];
}

bool comparePartitionResultIndex(const Value &a, const Value &b) {
    int64_t aIdx = getSequenceOutputIndex(a);
    int64_t bIdx = getSequenceOutputIndex(b);
    if (aIdx >= 0 && bIdx >= 0) {
        return aIdx < bIdx;
    }

    if (aIdx != bIdx) {
        return aIdx >= 0 && bIdx < 0;
    }

    auto aResult = llvm::dyn_cast<OpResult>(a);
    auto bResult = llvm::dyn_cast<OpResult>(b);
    if (aResult && bResult && a.getDefiningOp() == b.getDefiningOp()) {
        return aResult.getResultNumber() < bResult.getResultNumber();
    }

    return false;
}

bool isCompileTimeTosaConstant(Operation *op) { return llvm::isa_and_nonnull<tosa::ConstOp, tosa::ConstShapeOp>(op); }

bool isVulkanCustomShaderOperation(Operation *op) {
    auto customOp = llvm::dyn_cast_or_null<tosa::CustomOp>(op);
    return customOp && isVulkanCustomShaderOp(customOp);
}

struct PartitionDependencies {
    SmallVector<Value> inputs;
    SmallVector<Value> compileTimeConstantsToClone;
};

struct PartitionState {
    void addOp(int64_t partitionId, Operation *op) {
        partitionIdToOps[partitionId].push_back(op);
        highestPartitionId = std::max(highestPartitionId, partitionId);
    }

    void addResult(int64_t partitionId, Value value) { partitionIdToResults[partitionId].push_back(value); }

    void setResults(int64_t partitionId, SmallVector<Value> results) {
        partitionIdToResults[partitionId] = std::move(results);
    }

    SmallVector<Operation *> getOps(int64_t partitionId) const { return lookupValues(partitionIdToOps, partitionId); }

    SmallVector<Value> getResults(int64_t partitionId) const { return lookupValues(partitionIdToResults, partitionId); }

    bool hasPartitions() const { return highestPartitionId >= 0; }

    bool hasSinglePartition() const { return highestPartitionId == 0; }

    int64_t getLastPartitionId() const { return hasPartitions() ? highestPartitionId : 0; }

    void sortResults() {
        for (auto &entry : partitionIdToResults) {
            std::stable_sort(entry.second.begin(), entry.second.end(), comparePartitionResultIndex);
        }
    }

  private:
    template <typename T>
    static SmallVector<T> lookupValues(const DenseMap<int64_t, SmallVector<T>> &partitionValues, int64_t partitionId) {
        auto valuesIt = partitionValues.find(partitionId);
        if (valuesIt == partitionValues.end()) {
            return {};
        }
        return valuesIt->second;
    }

    int64_t highestPartitionId = -1;
    DenseMap<int64_t, SmallVector<Operation *>> partitionIdToOps;
    DenseMap<int64_t, SmallVector<Value>> partitionIdToResults;
};

struct SegmentPlan {
    int64_t partitionId;
    SmallVector<Operation *> ops;
    Operation *computeOp = nullptr;
    vgf::SegmentTypeEnum type = vgf::SegmentTypeEnum::GRAPH;
    SmallVector<Value> results;
    PartitionDependencies dependencies;
};

struct FunctionPartitionPlan {
    Operation *oldTerminator = nullptr;
    SmallVector<SegmentPlan, 8> segments;
};

struct PlannedFunctionPartition {
    func::FuncOp funcOp;
    FunctionPartitionPlan plan;
};

class FunctionPartitionPlanner {
  public:
    explicit FunctionPartitionPlanner(func::FuncOp funcOp)
        : funcOp(funcOp), oldTerminator(funcOp.getBody().front().getTerminator()) {}

    FunctionPartitionPlan collect();

  private:
    void collectPartitionState();
    SmallVector<SegmentPlan, 8> collectSegmentPlans();
    SegmentPlan collectSegmentPlan(int64_t partitionId, bool isPassthroughSegment);
    PartitionDependencies collectSegmentDependencies(const SegmentPlan &plan, bool isPassthroughSegment);
    PartitionDependencies collectPartitionDependencies(ArrayRef<Operation *> ops,
                                                       bool rematerializeCompileTimeConstants);
    SmallVector<Value> collectPassthroughSegmentResults();
    Operation *getComputeSegmentOp(ArrayRef<Operation *> partitionOps);
    bool hasExternalPartitionUse(Value value, int64_t partitionId);
    bool isPartitionResultValue(Operation *op, Value value, int64_t partitionId);
    bool hasRuntimeUseOfCompileTimeConstant(Operation *op);

    func::FuncOp funcOp;
    Operation *oldTerminator;
    PartitionState partitionState;
};

class FunctionPartitionEmitter {
  public:
    FunctionPartitionEmitter(OpBuilder &builder, func::FuncOp funcOp, Type segmentIdType)
        : builder(builder), funcOp(funcOp), segmentIdType(segmentIdType) {}

    void emit(const FunctionPartitionPlan &plan);

  private:
    struct CreatedSegment {
        std::string name;
        FunctionType functionType;
        vgf::SegmentOp op;
    };

    void removePartitioningAttrs(Operation *op);
    void markForDeletion(Operation *op);
    Operation *cloneWithoutPartitioningAttrs(Operation *op, IRMapping &mapping);
    Operation *clonePartitionOp(Operation *op, IRMapping &mapping);
    void cloneCompileTimeConstants(const SmallVector<Value> &constantsToClone, IRMapping &mapping);
    template <typename ArgumentsT>
    static void mapValuesToArguments(ArrayRef<Value> values, ArgumentsT arguments, IRMapping &mapping);
    static SmallVector<Value> lookupMappedValues(ArrayRef<Value> values, IRMapping &mapping);
    CreatedSegment createSegment(const SegmentPlan &plan);
    void createComputeSegmentBody(CreatedSegment &segment, const SegmentPlan &plan);
    void createGraphSegmentBody(CreatedSegment &segment, const SegmentPlan &plan);
    void createSegmentRun(CreatedSegment &segment, const SegmentPlan &plan);
    void emitSegment(const SegmentPlan &plan);

    OpBuilder &builder;
    func::FuncOp funcOp;
    Type segmentIdType;
    IRMapping externalMapping;
};

void FunctionPartitionPlanner::collectPartitionState() {
    partitionState = PartitionState();
    for (Operation &op : funcOp.getBody().front().without_terminator()) {
        auto partitionAttr = op.getAttrOfType<IntegerAttr>(graphPartitionIdAttrName);
        int64_t partitionId = partitionAttr.getInt();
        partitionState.addOp(partitionId, &op);

        auto leafAttr = op.getAttrOfType<BoolAttr>(graphPartitionLeafNodeAttrName);
        if (leafAttr.getValue()) {
            for (Value value : op.getResults()) {
                if (isPartitionResultValue(&op, value, partitionId)) {
                    partitionState.addResult(partitionId, value);
                }
            }
        }
    }
    partitionState.sortResults();
}

Operation *FunctionPartitionPlanner::getComputeSegmentOp(ArrayRef<Operation *> partitionOps) {
    if (partitionOps.size() != 1 || !isVulkanCustomShaderOperation(partitionOps.front())) {
        return nullptr;
    }
    return partitionOps.front();
}

bool FunctionPartitionPlanner::hasExternalPartitionUse(Value value, int64_t partitionId) {
    for (Operation *user : value.getUsers()) {
        if (llvm::isa<func::ReturnOp>(user)) {
            return true;
        }

        auto userPartitionAttr = user->getAttrOfType<IntegerAttr>(graphPartitionIdAttrName);
        if (userPartitionAttr && userPartitionAttr.getInt() != partitionId) {
            return true;
        }
    }
    return false;
}

bool FunctionPartitionPlanner::isPartitionResultValue(Operation *op, Value value, int64_t partitionId) {
    if (isCompileTimeTosaConstant(op) && !hasRuntimeUseOfCompileTimeConstant(op)) {
        return false;
    }

    return getSequenceOutputIndex(value) >= 0 || hasExternalPartitionUse(value, partitionId);
}

bool FunctionPartitionPlanner::hasRuntimeUseOfCompileTimeConstant(Operation *op) {
    return llvm::any_of(op->getUsers(), [](Operation *user) {
        return llvm::isa<func::ReturnOp>(user) || isVulkanCustomShaderOperation(user);
    });
}

PartitionDependencies FunctionPartitionPlanner::collectPartitionDependencies(ArrayRef<Operation *> ops,
                                                                             bool rematerializeCompileTimeConstants) {
    DenseSet<Operation *> knownOps(ops.begin(), ops.end());
    DenseSet<Value> seenRuntimeInputs;
    DenseSet<Value> seenConstantsToClone;
    PartitionDependencies dependencies;
    for (Operation *op : ops) {
        for (auto operand : op->getOperands()) {
            Operation *defOp = operand.getDefiningOp();
            if (knownOps.contains(defOp)) {
                continue;
            }

            if (rematerializeCompileTimeConstants && isCompileTimeTosaConstant(defOp)) {
                if (seenConstantsToClone.insert(operand).second) {
                    dependencies.compileTimeConstantsToClone.push_back(operand);
                }
                continue;
            }

            if (seenRuntimeInputs.insert(operand).second) {
                dependencies.inputs.push_back(operand);
            }
        }
    }

    return dependencies;
}

SmallVector<Value> FunctionPartitionPlanner::collectPassthroughSegmentResults() {
    DenseSet<Value> seenResults;
    SmallVector<Value> results;
    for (Value operand : oldTerminator->getOperands()) {
        if (seenResults.insert(operand).second) {
            results.push_back(operand);
        }
    }
    return results;
}

PartitionDependencies FunctionPartitionPlanner::collectSegmentDependencies(const SegmentPlan &plan,
                                                                           bool isPassthroughSegment) {
    PartitionDependencies dependencies;
    if (isPassthroughSegment) {
        dependencies.inputs.append(plan.results.begin(), plan.results.end());
    } else if (partitionState.hasSinglePartition()) {
        // This ensures that unused arguments are passed through when there is one segment(partition)
        auto args = funcOp.getArguments();
        std::copy(args.begin(), args.end(), std::back_inserter(dependencies.inputs));
    } else {
        dependencies = collectPartitionDependencies(plan.ops, plan.computeOp == nullptr);
    }

    return dependencies;
}

SegmentPlan FunctionPartitionPlanner::collectSegmentPlan(int64_t partitionId, bool isPassthroughSegment) {
    SegmentPlan plan;
    plan.partitionId = partitionId;
    plan.ops = partitionState.getOps(partitionId);
    plan.computeOp = getComputeSegmentOp(plan.ops);
    plan.type = plan.computeOp ? vgf::SegmentTypeEnum::COMPUTE : vgf::SegmentTypeEnum::GRAPH;
    plan.results = partitionState.getResults(partitionId);

    plan.dependencies = collectSegmentDependencies(plan, isPassthroughSegment);
    return plan;
}

SmallVector<SegmentPlan, 8> FunctionPartitionPlanner::collectSegmentPlans() {
    const bool isPassthroughSegment = !partitionState.hasPartitions();
    const int64_t lastPartitionId = partitionState.getLastPartitionId();
    if (isPassthroughSegment) {
        partitionState.setResults(0, collectPassthroughSegmentResults());
    }

    SmallVector<SegmentPlan, 8> segments;
    for (int64_t partitionId = 0; partitionId <= lastPartitionId; ++partitionId) {
        segments.push_back(collectSegmentPlan(partitionId, isPassthroughSegment));
    }
    return segments;
}

FunctionPartitionPlan FunctionPartitionPlanner::collect() {
    FunctionPartitionPlan plan;
    collectPartitionState();
    plan.oldTerminator = oldTerminator;
    plan.segments = collectSegmentPlans();
    return plan;
}

void FunctionPartitionEmitter::removePartitioningAttrs(Operation *op) {
    op->removeAttr(graphPartitionDeleteAttrName);
    op->removeAttr(graphPartitionIdAttrName);
    op->removeAttr(graphPartitionLeafNodeAttrName);
    op->removeAttr(graphPartitionSequenceOutputIndicesAttrName);
}

void FunctionPartitionEmitter::markForDeletion(Operation *op) {
    op->setAttr(graphPartitionDeleteAttrName, BoolAttr::get(builder.getContext(), true));
}

Operation *FunctionPartitionEmitter::cloneWithoutPartitioningAttrs(Operation *op, IRMapping &mapping) {
    Operation *clonedOp = builder.clone(*op, mapping);
    removePartitioningAttrs(clonedOp);
    return clonedOp;
}

Operation *FunctionPartitionEmitter::clonePartitionOp(Operation *op, IRMapping &mapping) {
    Operation *clonedOp = cloneWithoutPartitioningAttrs(op, mapping);
    markForDeletion(op);
    return clonedOp;
}

void FunctionPartitionEmitter::cloneCompileTimeConstants(const SmallVector<Value> &constantsToClone,
                                                         IRMapping &mapping) {
    for (Value operand : constantsToClone) {
        Operation *defOp = operand.getDefiningOp();
        Operation *clonedConst = cloneWithoutPartitioningAttrs(defOp, mapping);
        mapping.map(operand, clonedConst->getResult(0));
    }
}

template <typename ArgumentsT>
void FunctionPartitionEmitter::mapValuesToArguments(ArrayRef<Value> values, ArgumentsT arguments, IRMapping &mapping) {
    for (auto [value, argument] : llvm::zip(values, arguments)) {
        mapping.map(value, argument);
    }
}

SmallVector<Value> FunctionPartitionEmitter::lookupMappedValues(ArrayRef<Value> values, IRMapping &mapping) {
    return llvm::map_to_vector(values, [&](Value value) { return mapping.lookupOrDefault(value); });
}

void deleteOldOps(ModuleOp moduleOp) {
    std::vector<Operation *> opsToDelete;
    moduleOp.walk([&](Operation *op) {
        if (auto attr = op->getAttrOfType<BoolAttr>(graphPartitionDeleteAttrName)) {
            if (attr.getValue()) {
                opsToDelete.push_back(op);
            }
        }
    });
    for (auto it = opsToDelete.rbegin(); it != opsToDelete.rend(); ++it) {
        (*it)->erase();
    }
}

FunctionPartitionEmitter::CreatedSegment FunctionPartitionEmitter::createSegment(const SegmentPlan &plan) {
    CreatedSegment segment;
    segment.name = "graph_partition_" + std::to_string(plan.partitionId);
    segment.functionType =
        builder.getFunctionType(ValueRange(plan.dependencies.inputs).getTypes(), ValueRange(plan.results).getTypes());
    segment.op = vgf::SegmentOp::create(builder, funcOp.getLoc(), segment.name, plan.type, segment.functionType,
                                        nullptr, nullptr);
    segment.op->setAttr("segment_id", IntegerAttr::get(segmentIdType, plan.partitionId));
    return segment;
}

void FunctionPartitionEmitter::createComputeSegmentBody(CreatedSegment &segment, const SegmentPlan &plan) {
    OpBuilder::InsertionGuard segmentGuard{builder};
    Block *segmentBlock = segment.op.addEntryBlock();
    builder.setInsertionPoint(segmentBlock, segmentBlock->end());

    IRMapping segmentMapping;
    mapValuesToArguments(plan.dependencies.inputs, segment.op.getArguments(), segmentMapping);

    clonePartitionOp(plan.computeOp, segmentMapping);

    SmallVector<Value> segmentResults = lookupMappedValues(plan.results, segmentMapping);
    vgf::SegmentOutputOp::create(builder, segment.op.getLoc(), segmentResults);
}

void FunctionPartitionEmitter::createGraphSegmentBody(CreatedSegment &segment, const SegmentPlan &plan) {
    OpBuilder::InsertionGuard segmentGuard{builder};
    Block *segmentBlock = segment.op.addEntryBlock();
    builder.setInsertionPoint(segmentBlock, segmentBlock->end());

    auto newFuncOp = func::FuncOp::create(builder, funcOp.getLoc(), segment.name, segment.functionType);
    newFuncOp->setAttr("segment_id", IntegerAttr::get(segmentIdType, plan.partitionId));

    llvm::SmallVector<Value, 0> segmentResults;
    vgf::SegmentOutputOp::create(builder, segment.op.getLoc(), segmentResults);

    {
        OpBuilder::InsertionGuard funcGuard{builder};
        Block *funcBlock = newFuncOp.addEntryBlock();
        builder.setInsertionPoint(funcBlock, funcBlock->end());
        IRMapping funcMapping;
        mapValuesToArguments(plan.dependencies.inputs, newFuncOp.getArguments(), funcMapping);

        cloneCompileTimeConstants(plan.dependencies.compileTimeConstantsToClone, funcMapping);
        for (Operation *op : plan.ops) {
            clonePartitionOp(op, funcMapping);
        }

        SmallVector<Value> funcResults = lookupMappedValues(plan.results, funcMapping);
        func::ReturnOp::create(builder, newFuncOp.getLoc(), funcResults);
    }
}

void FunctionPartitionEmitter::createSegmentRun(CreatedSegment &segment, const SegmentPlan &plan) {
    SmallVector<Value> runInputs = lookupMappedValues(plan.dependencies.inputs, externalMapping);
    auto segmentRunOp = vgf::SegmentRunOp::create(builder, segment.op.getLoc(), segment.op.getResultTypes(),
                                                  SymbolRefAttr::get(segment.op), ValueRange(runInputs));
    segmentRunOp->setAttr("segment_id", IntegerAttr::get(segmentIdType, plan.partitionId));

    for (auto [result, segmentRunOpArg] : llvm::zip(plan.results, segmentRunOp.getResults())) {
        externalMapping.map(result, segmentRunOpArg);
    }
}

void FunctionPartitionEmitter::emitSegment(const SegmentPlan &plan) {
    CreatedSegment segment = createSegment(plan);

    if (plan.computeOp) {
        createComputeSegmentBody(segment, plan);
    } else {
        createGraphSegmentBody(segment, plan);
    }
    createSegmentRun(segment, plan);
}

void FunctionPartitionEmitter::emit(const FunctionPartitionPlan &plan) {
    for (const SegmentPlan &segment : plan.segments) {
        emitSegment(segment);
    }

    builder.clone(*plan.oldTerminator, externalMapping);
    markForDeletion(plan.oldTerminator);
}

std::vector<PlannedFunctionPartition> collectFunctionPartitionPlans(ModuleOp moduleOp) {
    std::vector<PlannedFunctionPartition> plannedFunctions;
    for (func::FuncOp funcOp : moduleOp.getOps<func::FuncOp>()) {
        PlannedFunctionPartition plannedFunction;
        plannedFunction.funcOp = funcOp;
        plannedFunction.plan = FunctionPartitionPlanner(funcOp).collect();
        plannedFunctions.push_back(std::move(plannedFunction));
    }
    return plannedFunctions;
}

void emitFunctionPartitionPlans(ArrayRef<PlannedFunctionPartition> plannedFunctions, MLIRContext *context) {
    OpBuilder builder(context);
    const Type segmentIdType = IntegerType::get(context, 32, IntegerType::SignednessSemantics::Unsigned);

    for (const PlannedFunctionPartition &plannedFunction : plannedFunctions) {
        builder.setInsertionPoint(plannedFunction.plan.oldTerminator);
        FunctionPartitionEmitter emitter(builder, plannedFunction.funcOp, segmentIdType);
        emitter.emit(plannedFunction.plan);
    }
}

void partitionModuleFunctions(ModuleOp moduleOp, MLIRContext *context) {
    std::vector<PlannedFunctionPartition> plannedFunctions = collectFunctionPartitionPlans(moduleOp);
    emitFunctionPartitionPlans(plannedFunctions, context);
}

LogicalResult convertPartitionedModule(ModuleOp moduleOp, MLIRContext *context, bool analysis) {
    ConversionTarget target(*context);
    target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) { return !llvm::isa<ModuleOp>(op->getParentOp()); });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [](func::ReturnOp op) { return !llvm::isa<ModuleOp>(op->getParentOp()->getParentOp()); });
    target.addDynamicallyLegalOp<tosa::CustomOp>([](tosa::CustomOp op) { return !isVulkanCustomShaderOp(op); });
    target.addLegalDialect<vgf::VGFDialect, func::FuncDialect>();

    RewritePatternSet patterns(context);
    patterns.add<FuncOpRewriter, ReturnOpRewriter>(context);
    patterns.add<TosaCustomOpRewriter>(context, analysis);
    return applyPartialConversion(moduleOp, target, std::move(patterns));
}

bool hasFunctionDeclaration(ModuleOp moduleOp) {
    for (func::FuncOp funcOp : moduleOp.getOps<func::FuncOp>()) {
        if (funcOp.isDeclaration()) {
            funcOp.emitError("model partitioning requires function definitions");
            return true;
        }
    }
    return false;
}

class ModelPartitioningPass : public impl::ModelPartitioningPassBase<ModelPartitioningPass> {
  public:
    using impl::ModelPartitioningPassBase<ModelPartitioningPass>::ModelPartitioningPassBase;

    void runOnOperation() override {
        mlir::ModuleOp moduleOp = getOperation();
        MLIRContext *context = &getContext();

        if (hasFunctionDeclaration(moduleOp)) {
            return signalPassFailure();
        }

        partitionModuleFunctions(moduleOp, context);
        deleteOldOps(moduleOp);

        if (convertPartitionedModule(moduleOp, context, analysis).failed()) {
            return signalPassFailure();
        }
    }
};

} // namespace

} // namespace mlir::model_converter_passes
