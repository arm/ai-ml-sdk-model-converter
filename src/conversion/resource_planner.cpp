/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "conversion/resource_planner.hpp"

#include "utils.hpp"

#define VGFLIB_VK_HELPERS
#include "vgf/vulkan_helpers.generated.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <iterator>

namespace mlir::model_converter_passes::detail {
namespace {

// As defined in vulkan_core.h
constexpr DescriptorType DESCRIPTOR_TYPE_TENSOR_ARM = 1000460000;
constexpr StringRef UNSIGNED_INPUT_OUTPUT_ATTR = "mlsdk.unsigned_input_output";
constexpr StringRef MIN_FILTER_ATTR = "min_filter";
constexpr StringRef MAG_FILTER_ATTR = "mag_filter";
constexpr StringRef ADDRESS_MODE_U_ATTR = "address_mode_u";
constexpr StringRef ADDRESS_MODE_V_ATTR = "address_mode_v";
constexpr StringRef BORDER_COLOR_ATTR = "border_color";

template <typename ParseFn> uint32_t parseSamplerEnumValue(const StringRef name, ParseFn &&parseFn) {
    if (name.empty()) {
        return UNSET_SAMPLER_VALUE;
    }
    const auto value = parseFn(name.str());
    return value < 0 ? UNSET_SAMPLER_VALUE : static_cast<uint32_t>(value);
}

uint32_t parseSamplerFilter(const StringRef name) { return parseSamplerEnumValue(name, NameToFilterType); }

uint32_t parseSamplerAddressMode(const StringRef name) {
    return parseSamplerEnumValue(name, NameToSamplerAddressModeType);
}

uint32_t parseSamplerBorderColor(const StringRef name) { return parseSamplerEnumValue(name, NameToBorderColorType); }

DictionaryAttr getSamplerConfig(ArrayAttr samplerConfigsAttr, size_t index) {
    if (!samplerConfigsAttr || index >= samplerConfigsAttr.size()) {
        return nullptr;
    }
    return llvm::dyn_cast<DictionaryAttr>(samplerConfigsAttr[static_cast<unsigned>(index)]);
}

uint32_t getSamplerValue(DictionaryAttr samplerConfigAttr, StringRef key,
                         const std::function<uint32_t(StringRef)> &parseFn) {
    if (!samplerConfigAttr) {
        return UNSET_SAMPLER_VALUE;
    }
    auto valueAttr = samplerConfigAttr.getAs<StringAttr>(key);
    if (!valueAttr) {
        return UNSET_SAMPLER_VALUE;
    }
    return parseFn(valueAttr.getValue());
}

std::optional<SamplerConfigValues> getOptionalSamplerConfigValues(ArrayAttr samplerConfigsAttr, size_t index) {
    const auto samplerConfigAttr = getSamplerConfig(samplerConfigsAttr, index);
    if (!samplerConfigAttr || samplerConfigAttr.empty()) {
        return std::nullopt;
    }

    SamplerConfigValues samplerConfig;
    samplerConfig.minFilter = getSamplerValue(samplerConfigAttr, MIN_FILTER_ATTR, parseSamplerFilter);
    samplerConfig.magFilter = getSamplerValue(samplerConfigAttr, MAG_FILTER_ATTR, parseSamplerFilter);
    samplerConfig.addressModeU = getSamplerValue(samplerConfigAttr, ADDRESS_MODE_U_ATTR, parseSamplerAddressMode);
    samplerConfig.addressModeV = getSamplerValue(samplerConfigAttr, ADDRESS_MODE_V_ATTR, parseSamplerAddressMode);
    samplerConfig.borderColor = getSamplerValue(samplerConfigAttr, BORDER_COLOR_ATTR, parseSamplerBorderColor);
    return samplerConfig;
}

const std::vector<BindingSlotRef> &
getSegmentBindings(const std::map<SegmentId, std::vector<BindingSlotRef>> &bindingsBySegment, SegmentId segmentId) {
    static const std::vector<BindingSlotRef> emptyBindings = {};
    auto bindingsIt = bindingsBySegment.find(segmentId);
    if (bindingsIt == bindingsBySegment.end()) {
        return emptyBindings;
    }
    return bindingsIt->second;
}

} // namespace

ResourcePlanner::ResourcePlanner(vgf::SequenceOp &sequenceOp)
    : _sequenceOp(sequenceOp), _sequenceOutputOp(sequenceOp.front().getTerminator()),
      _bindingId(sequenceOp.getNumArguments()) {
    _resourcePlan.sequenceInputValues.assign(sequenceOp.getArguments().begin(), sequenceOp.getArguments().end());
    _resourcePlan.sequenceOutputValues.assign(_sequenceOutputOp->getOperands().begin(),
                                              _sequenceOutputOp->getOperands().end());
    for (const auto &[outputIndex, outputValue] : llvm::enumerate(_sequenceOutputOp->getOperands())) {
        _sequenceOutputIndices[outputValue] = static_cast<uint32_t>(outputIndex);
    }
}

LogicalResult ResourcePlanner::buildPlan() {
    if (collectSequenceInputs().failed() || collectIntermediates().failed() || collectSequenceOutputs().failed()) {
        return failure();
    }

    finalizePlannedValues();
    return success();
}

const ResourcePlan &ResourcePlanner::getPlan() const { return _resourcePlan; }

bool ResourcePlanner::isSequenceInputOperand(Value operand) const {
    return std::any_of(_sequenceOp.getArguments().begin(), _sequenceOp.getArguments().end(),
                       [&](const Value sequenceOpArgument) { return sequenceOpArgument == operand; });
}

bool ResourcePlanner::isSequenceOutputOperand(Value operand) const {
    return _sequenceOutputIndices.find(operand) != _sequenceOutputIndices.end();
}

bool ResourcePlanner::getSequenceInputUnsigned(uint32_t inputIndex) const {
    if (auto unsignedAttr = _sequenceOp.getArgAttrOfType<BoolAttr>(inputIndex, UNSIGNED_INPUT_OUTPUT_ATTR)) {
        return unsignedAttr.getValue();
    }
    return false;
}

bool ResourcePlanner::getSequenceOutputUnsigned(uint32_t outputIndex) const {
    if (auto unsignedAttr = _sequenceOp.getResultAttrOfType<BoolAttr>(outputIndex, UNSIGNED_INPUT_OUTPUT_ATTR)) {
        return unsignedAttr.getValue();
    }
    return false;
}

FailureOr<std::vector<int64_t>> ResourcePlanner::getValueShape(Value value) const {
    const ShapedType type = (isSequenceInputOperand(value) || isSequenceOutputOperand(value))
                                ? llvm::dyn_cast<ShapedType>(value.getType())
                                : convertShapedType(value.getType());
    if (!type) {
        return _sequenceOp.emitError("expected shaped value when serializing VGF resources");
    }
    return std::vector<int64_t>(type.getShape().begin(), type.getShape().end());
}

LogicalResult ResourcePlanner::getGraphView(Value value, StringRef role, StringRef segmentName,
                                            ResourceViewKey &view) const {
    VGFBuilder::VkFormat vkFormat;
    const ShapedType type = (isSequenceInputOperand(value) || isSequenceOutputOperand(value))
                                ? llvm::dyn_cast<ShapedType>(value.getType())
                                : convertShapedType(value.getType());
    if (!type) {
        return _sequenceOp.emitError("expected shaped value for ") << role << " in segment " << segmentName;
    }

    bool isUnsigned = false;
    if (isSequenceInputOperand(value)) {
        isUnsigned = getSequenceInputUnsigned(llvm::cast<BlockArgument>(value).getArgNumber());
    } else if (isSequenceOutputOperand(value)) {
        auto outputIndexIt = _sequenceOutputIndices.find(value);
        if (outputIndexIt == _sequenceOutputIndices.end()) {
            return _sequenceOp.emitError("failed to resolve sequence output index for ")
                   << role << " in segment " << segmentName;
        }
        isUnsigned = getSequenceOutputUnsigned(outputIndexIt->second);
    }

    if (VGFBuilder::mlirTypeToVkFormat(type.getElementType(), vkFormat, isUnsigned).failed()) {
        return _sequenceOp.emitError("unsupported type for ") << role << " in segment " << segmentName;
    }

    view = {DESCRIPTOR_TYPE_TENSOR_ARM, static_cast<FormatType>(vkFormat)};
    return success();
}

ResourceKey ResourcePlanner::makeResourceKey(ResourceCategory category, const ResourceViewKey &view,
                                             std::optional<SamplerConfigValues> samplerConfig) {
    return ResourceKey{category, view, samplerConfig};
}

FailureOr<PlannedValue *> ResourcePlanner::ensurePlannedValue(Value value, std::optional<uint32_t> bindingIndex) {
    auto [it, inserted] = _resourcePlan.plannedValues.try_emplace(value);
    auto &plan = it->second;
    if (inserted) {
        auto shape = getValueShape(value);
        if (failed(shape)) {
            _resourcePlan.plannedValues.erase(it);
            return failure();
        }
        plan.shape = std::move(*shape);
    }

    if (bindingIndex.has_value()) {
        if (!plan.bindingAssigned) {
            plan.bindingIndex = *bindingIndex;
            plan.bindingAssigned = true;
        } else if (plan.bindingIndex != *bindingIndex) {
            return _sequenceOp.emitError("logical value changed VGF binding index");
        }
    }

    return &plan;
}

void ResourcePlanner::appendResourceRequirement(PlannedValue &plan, const ResourceKey &resourceKey, bool isProducer) {
    if (std::find(plan.resourceOrder.begin(), plan.resourceOrder.end(), resourceKey) == plan.resourceOrder.end()) {
        plan.resourceOrder.push_back(resourceKey);
    }
    if (isProducer) {
        plan.producerResource = resourceKey;
    }
}

void ResourcePlanner::appendAttachment(std::vector<SegmentAttachment> &attachments, SegmentId segmentId, Value value,
                                       bool isOutput, const ResourceKey &resourceKey,
                                       std::optional<int64_t> descriptorSet, std::optional<uint32_t> descriptorBinding,
                                       bool isProducer) {
    auto planIt = _resourcePlan.plannedValues.find(value);
    assert(planIt != _resourcePlan.plannedValues.end());
    appendResourceRequirement(planIt->second, resourceKey, isProducer);
    attachments.push_back({segmentId, value, isOutput, resourceKey, descriptorSet, descriptorBinding});
}

LogicalResult ResourcePlanner::collectSequenceInputs() {
    for (auto operand : _sequenceOp.getArguments()) {
        if (failed(ensurePlannedValue(operand, llvm::cast<BlockArgument>(operand).getArgNumber()))) {
            return failure();
        }

        const WalkResult inputWalkResult = _sequenceOp.walk([&](vgf::SegmentOp segmentOp) {
            const auto segmentType = segmentOp.getSegmentType();
            const auto segmentId = segmentOp->getAttrOfType<IntegerAttr>("segment_id").getUInt();
            const auto segmentName = segmentOp.getSymName().str();
            auto *runSegmentOp = segmentOp->getNextNode();

            WalkResult segmentWalkResult;
            if (segmentType == vgf::SegmentTypeEnum::COMPUTE) {
                segmentWalkResult = segmentOp.walk([&](vgf::ShaderPlaceholderOp shaderPlaceholderOp) {
                    size_t ioIndex = 0;
                    for (const auto &[inputBinding, inputDescriptorType, inputVkFormat, inputDescriptorSet, input] :
                         llvm::zip(shaderPlaceholderOp.getInputBindings(),
                                   shaderPlaceholderOp.getInputVkDescriptorTypes(),
                                   shaderPlaceholderOp.getInputVkFormats(),
                                   shaderPlaceholderOp.getInputDescriptorSets(), runSegmentOp->getOperands())) {
                        if (input == operand) {
                            const ResourceViewKey view = {
                                NameToDescriptorType(llvm::cast<StringAttr>(inputDescriptorType).str()),
                                NameToFormatType(llvm::cast<StringAttr>(inputVkFormat).str())};
                            appendAttachment(
                                _resourcePlan.sequenceInputAttachments, segmentId, input, false,
                                makeResourceKey(ResourceCategory::INPUT, view,
                                                getOptionalSamplerConfigValues(
                                                    shaderPlaceholderOp.getInputSamplerConfigsAttr(), ioIndex)),
                                inputDescriptorSet, static_cast<uint32_t>(inputBinding));
                        }
                        ++ioIndex;
                    }

                    return WalkResult::advance();
                });
            } else if (segmentType == vgf::SegmentTypeEnum::GRAPH) {
                segmentWalkResult = segmentOp.walk([&](spirv::ModuleOp spirvModuleOp) {
                    WalkResult spirvModuleWalkResult = spirvModuleOp.walk([&](spirv::GraphARMOp) {
                        for (const auto &input : runSegmentOp->getOperands()) {
                            if (input == operand) {
                                ResourceViewKey view;
                                if (getGraphView(input, "input", segmentName, view).failed()) {
                                    return WalkResult::interrupt();
                                }
                                appendAttachment(_resourcePlan.sequenceInputAttachments, segmentId, input, false,
                                                 makeResourceKey(ResourceCategory::INPUT, view));
                            }
                        }

                        return WalkResult::advance();
                    });
                    return spirvModuleWalkResult;
                });
            } else {
                llvm::errs() << "Invalid segment type in sequence module\n";
                return WalkResult::interrupt();
            }
            return segmentWalkResult;
        });

        if (inputWalkResult.wasInterrupted()) {
            return failure();
        }
    }

    return success();
}

LogicalResult ResourcePlanner::collectIntermediates() {
    const WalkResult intermediateWalkResult = _sequenceOp.walk([&](vgf::SegmentOp segmentOp) {
        const auto segmentType = segmentOp.getSegmentType();
        const auto segmentId = segmentOp->getAttrOfType<IntegerAttr>("segment_id").getUInt();
        const auto segmentName = segmentOp.getSymName().str();
        auto *runSegmentOp = segmentOp->getNextNode();

        WalkResult segmentWalkResult;
        if (segmentType == vgf::SegmentTypeEnum::COMPUTE) {
            segmentWalkResult = segmentOp.walk([&](vgf::ShaderPlaceholderOp shaderPlaceholderOp) {
                size_t ioIndex = 0;
                for (const auto &[inputBinding, inputDescriptorType, inputVkFormat, inputDescriptorSet, input] :
                     llvm::zip(shaderPlaceholderOp.getInputBindings(), shaderPlaceholderOp.getInputVkDescriptorTypes(),
                               shaderPlaceholderOp.getInputVkFormats(), shaderPlaceholderOp.getInputDescriptorSets(),
                               runSegmentOp->getOperands())) {
                    if (!isSequenceInputOperand(input)) {
                        auto plan = ensurePlannedValue(input);
                        if (failed(plan)) {
                            return WalkResult::interrupt();
                        }
                        auto &plannedValue = **plan;
                        if (!plannedValue.bindingAssigned) {
                            plannedValue.bindingIndex = _bindingId++;
                            plannedValue.bindingAssigned = true;
                            _resourcePlan.intermediateValues.push_back(input);
                        }

                        const ResourceViewKey view = {
                            NameToDescriptorType(llvm::cast<StringAttr>(inputDescriptorType).str()),
                            NameToFormatType(llvm::cast<StringAttr>(inputVkFormat).str())};
                        appendAttachment(
                            _resourcePlan.intermediateAttachments, segmentId, input, false,
                            makeResourceKey(ResourceCategory::INTERMEDIATE, view,
                                            getOptionalSamplerConfigValues(
                                                shaderPlaceholderOp.getInputSamplerConfigsAttr(), ioIndex)),
                            inputDescriptorSet, static_cast<uint32_t>(inputBinding));
                    }
                    ++ioIndex;
                }

                ioIndex = 0;
                for (const auto &[outputBinding, outputDescriptorType, outputVkFormat, outputDescriptorSet, result] :
                     llvm::zip(shaderPlaceholderOp.getOutputBindings(),
                               shaderPlaceholderOp.getOutputVkDescriptorTypes(),
                               shaderPlaceholderOp.getOutputVkFormats(), shaderPlaceholderOp.getOutputDescriptorSets(),
                               runSegmentOp->getResults())) {
                    if (!isSequenceOutputOperand(result)) {
                        auto plan = ensurePlannedValue(result);
                        if (failed(plan)) {
                            return WalkResult::interrupt();
                        }
                        auto &plannedValue = **plan;
                        if (!plannedValue.bindingAssigned) {
                            plannedValue.bindingIndex = _bindingId++;
                            plannedValue.bindingAssigned = true;
                            _resourcePlan.intermediateValues.push_back(result);
                        }

                        const ResourceViewKey view = {
                            NameToDescriptorType(llvm::cast<StringAttr>(outputDescriptorType).str()),
                            NameToFormatType(llvm::cast<StringAttr>(outputVkFormat).str())};
                        appendAttachment(
                            _resourcePlan.intermediateAttachments, segmentId, result, true,
                            makeResourceKey(ResourceCategory::INTERMEDIATE, view,
                                            getOptionalSamplerConfigValues(
                                                shaderPlaceholderOp.getOutputSamplerConfigsAttr(), ioIndex)),
                            outputDescriptorSet, static_cast<uint32_t>(outputBinding), true);
                    }
                    ++ioIndex;
                }
                return WalkResult::advance();
            });
        } else if (segmentType == vgf::SegmentTypeEnum::GRAPH) {
            segmentWalkResult = segmentOp.walk([&](spirv::ModuleOp spirvModuleOp) {
                WalkResult spirvModuleWalkResult = spirvModuleOp.walk([&](spirv::GraphARMOp) {
                    for (const auto &input : runSegmentOp->getOperands()) {
                        if (!isSequenceInputOperand(input)) {
                            ResourceViewKey view;
                            if (getGraphView(input, "input", segmentName, view).failed()) {
                                return WalkResult::interrupt();
                            }

                            auto plan = ensurePlannedValue(input);
                            if (failed(plan)) {
                                return WalkResult::interrupt();
                            }
                            auto &plannedValue = **plan;
                            if (!plannedValue.bindingAssigned) {
                                plannedValue.bindingIndex = _bindingId++;
                                plannedValue.bindingAssigned = true;
                                _resourcePlan.intermediateValues.push_back(input);
                            }

                            appendAttachment(_resourcePlan.intermediateAttachments, segmentId, input, false,
                                             makeResourceKey(ResourceCategory::INTERMEDIATE, view));
                        }
                    }

                    for (const auto &result : runSegmentOp->getResults()) {
                        if (!isSequenceOutputOperand(result)) {
                            ResourceViewKey view;
                            if (getGraphView(result, "result", segmentName, view).failed()) {
                                return WalkResult::interrupt();
                            }

                            auto plan = ensurePlannedValue(result);
                            if (failed(plan)) {
                                return WalkResult::interrupt();
                            }
                            auto &plannedValue = **plan;
                            if (!plannedValue.bindingAssigned) {
                                plannedValue.bindingIndex = _bindingId++;
                                plannedValue.bindingAssigned = true;
                                _resourcePlan.intermediateValues.push_back(result);
                            }

                            appendAttachment(_resourcePlan.intermediateAttachments, segmentId, result, true,
                                             makeResourceKey(ResourceCategory::INTERMEDIATE, view), std::nullopt,
                                             std::nullopt, true);
                        }
                    }
                    return WalkResult::advance();
                });
                return spirvModuleWalkResult;
            });
        } else {
            return WalkResult::interrupt();
        }
        return segmentWalkResult;
    });

    return success(!intermediateWalkResult.wasInterrupted());
}

bool ResourcePlanner::hasResourceCategory(const PlannedValue &plan, ResourceCategory category) const {
    return std::any_of(plan.resourceOrder.begin(), plan.resourceOrder.end(),
                       [&](const ResourceKey &key) { return key.category == category; });
}

std::optional<ResourceKey> ResourcePlanner::getCanonicalResource(const PlannedValue &plan,
                                                                 ResourceCategory category) const {
    if (plan.producerResource.has_value() && plan.producerResource->category == category) {
        return plan.producerResource;
    }
    auto resourceIt = std::find_if(plan.resourceOrder.begin(), plan.resourceOrder.end(),
                                   [&](const ResourceKey &key) { return key.category == category; });
    if (resourceIt == plan.resourceOrder.end()) {
        return std::nullopt;
    }
    return *resourceIt;
}

LogicalResult ResourcePlanner::collectSequenceOutputs() {
    for (auto operand : _sequenceOutputOp->getOperands()) {
        auto plan = ensurePlannedValue(operand);
        if (failed(plan)) {
            return failure();
        }
        auto &plannedValue = **plan;
        if (!plannedValue.bindingAssigned) {
            plannedValue.bindingIndex = _bindingId++;
            plannedValue.bindingAssigned = true;
        }

        const WalkResult outputWalkResult = _sequenceOp.walk([&](vgf::SegmentOp segmentOp) {
            const auto segmentType = segmentOp.getSegmentType();
            const auto segmentId = segmentOp->getAttrOfType<IntegerAttr>("segment_id").getUInt();
            const auto segmentName = segmentOp.getSymName().str();
            auto *runSegmentOp = segmentOp->getNextNode();

            WalkResult segmentWalkResult;
            if (segmentType == vgf::SegmentTypeEnum::COMPUTE) {
                segmentWalkResult = segmentOp.walk([&](vgf::ShaderPlaceholderOp shaderPlaceholderOp) {
                    size_t ioIndex = 0;
                    for (const auto &[outputBinding, outputDescriptorType, outputVkFormat, outputDescriptorSet,
                                      result] :
                         llvm::zip(shaderPlaceholderOp.getOutputBindings(),
                                   shaderPlaceholderOp.getOutputVkDescriptorTypes(),
                                   shaderPlaceholderOp.getOutputVkFormats(),
                                   shaderPlaceholderOp.getOutputDescriptorSets(), runSegmentOp->getResults())) {
                        if (result == operand) {
                            const ResourceViewKey view = {
                                NameToDescriptorType(llvm::cast<StringAttr>(outputDescriptorType).str()),
                                NameToFormatType(llvm::cast<StringAttr>(outputVkFormat).str())};
                            appendAttachment(
                                _resourcePlan.sequenceOutputAttachments, segmentId, result, true,
                                makeResourceKey(ResourceCategory::OUTPUT, view,
                                                getOptionalSamplerConfigValues(
                                                    shaderPlaceholderOp.getOutputSamplerConfigsAttr(), ioIndex)),
                                outputDescriptorSet, static_cast<uint32_t>(outputBinding), true);
                        }
                        ++ioIndex;
                    }

                    return WalkResult::advance();
                });
            } else if (segmentType == vgf::SegmentTypeEnum::GRAPH) {
                segmentWalkResult = segmentOp.walk([&](spirv::ModuleOp spirvModuleOp) {
                    WalkResult spirvModuleWalkResult = spirvModuleOp.walk([&](spirv::GraphARMOp) {
                        for (const auto &result : runSegmentOp->getResults()) {
                            if (result == operand) {
                                ResourceViewKey view;
                                if (getGraphView(result, "result", segmentName, view).failed()) {
                                    return WalkResult::interrupt();
                                }
                                appendAttachment(_resourcePlan.sequenceOutputAttachments, segmentId, result, true,
                                                 makeResourceKey(ResourceCategory::OUTPUT, view), std::nullopt,
                                                 std::nullopt, true);
                            }
                        }

                        return WalkResult::advance();
                    });
                    return spirvModuleWalkResult;
                });
            } else {
                return WalkResult::interrupt();
            }
            return segmentWalkResult;
        });

        if (outputWalkResult.wasInterrupted()) {
            return failure();
        }

        if (!hasResourceCategory(plannedValue, ResourceCategory::OUTPUT)) {
            ResourceViewKey view;
            if (getGraphView(operand, "result", "sequence_output", view).failed()) {
                return failure();
            }
            appendResourceRequirement(plannedValue, makeResourceKey(ResourceCategory::OUTPUT, view), true);
        }
    }

    return success();
}

void ResourcePlanner::finalizePlannedValues() {
    for (auto operand : _sequenceOp.getArguments()) {
        finalizePlannedValue(operand);
    }
    for (auto value : _resourcePlan.intermediateValues) {
        finalizePlannedValue(value);
    }
    for (auto operand : _sequenceOutputOp->getOperands()) {
        finalizePlannedValue(operand);
    }
}

void ResourcePlanner::finalizePlannedValue(Value value) {
    auto planIt = _resourcePlan.plannedValues.find(value);
    if (planIt == _resourcePlan.plannedValues.end()) {
        return;
    }

    auto &plan = planIt->second;
    if (plan.resourceOrder.empty()) {
        return;
    }

    if (plan.resourceOrder.size() > 1 && !plan.aliasGroupId.has_value()) {
        assert(_nextAliasGroupId != INVALID_ALIAS_GROUP_ID && "exhausted alias group ids");
        plan.aliasGroupId = _nextAliasGroupId++;
    }
}

ResourcePlanEncoder::ResourcePlanEncoder(const ResourcePlan &resourcePlan, VGFBuilder &vgfBuilder)
    : _resourcePlan(resourcePlan), _vgfBuilder(vgfBuilder) {}

const std::vector<BindingSlotRef> &EncodedResourcePlan::getSegmentInputBindings(SegmentId segmentId) const {
    return getSegmentBindings(segmentInputBindings, segmentId);
}

const std::vector<BindingSlotRef> &EncodedResourcePlan::getSegmentOutputBindings(SegmentId segmentId) const {
    return getSegmentBindings(segmentOutputBindings, segmentId);
}

const EncodedResourcePlan &ResourcePlanEncoder::encode() {
    materializeBindings();
    return _encodedPlan;
}

std::optional<ResourceKey> ResourcePlanEncoder::getCanonicalResource(const PlannedValue &plan,
                                                                     ResourceCategory category) {
    if (plan.producerResource.has_value() && plan.producerResource->category == category) {
        return plan.producerResource;
    }
    auto resourceIt = std::find_if(plan.resourceOrder.begin(), plan.resourceOrder.end(),
                                   [&](const ResourceKey &key) { return key.category == category; });
    if (resourceIt == plan.resourceOrder.end()) {
        return std::nullopt;
    }
    return *resourceIt;
}

void ResourcePlanEncoder::addBindingToDescriptorSet(SegmentId segmentId, int64_t descriptorSet,
                                                    BindingSlotRef bindingSlotRef) {
    const auto descriptorSetIndex = static_cast<uint32_t>(descriptorSet);
    auto &bindingRefs = _segmentDescriptorSetBindingRefs[segmentId][descriptorSetIndex];
    if (bindingRefs.insert(bindingSlotRef.reference).second) {
        _encodedPlan.segmentDescriptorSetBindings[segmentId][descriptorSetIndex].push_back(bindingSlotRef);
    }
}

ResourceRef ResourcePlanEncoder::createResource(const PlannedValue &plan, const ResourceKey &resourceKey) {
    switch (resourceKey.category) {
    case ResourceCategory::INPUT:
        return _vgfBuilder.getEncoder()->AddInputResource(resourceKey.view.descriptorType, resourceKey.view.vkFormat,
                                                          plan.shape, {}, plan.aliasGroupId);
    case ResourceCategory::OUTPUT:
        return _vgfBuilder.getEncoder()->AddOutputResource(resourceKey.view.descriptorType, resourceKey.view.vkFormat,
                                                           plan.shape, {}, plan.aliasGroupId);
    case ResourceCategory::INTERMEDIATE:
        return _vgfBuilder.getEncoder()->AddIntermediateResource(
            resourceKey.view.descriptorType, resourceKey.view.vkFormat, plan.shape, {}, plan.aliasGroupId);
    case ResourceCategory::CONSTANT:
        assert(false && "planned VGF resources must not be constants");
        return ResourceRef{0};
    }
    assert(false && "invalid planned VGF resource category");
    return ResourceRef{0};
}

void ResourcePlanEncoder::addSamplerConfig(ResourceRef resourceRef, const ResourceKey &resourceKey) {
    if (!resourceKey.samplerConfig.has_value()) {
        return;
    }

    const auto &samplerConfig = *resourceKey.samplerConfig;
    _vgfBuilder.getEncoder()->AddSamplerConfig(resourceRef, samplerConfig.minFilter, samplerConfig.magFilter,
                                               samplerConfig.addressModeU, samplerConfig.addressModeV,
                                               samplerConfig.borderColor);
}

ResourceRef ResourcePlanEncoder::getOrCreateResource(Value value, const ResourceKey &resourceKey) {
    auto planIt = _resourcePlan.plannedValues.find(value);
    assert(planIt != _resourcePlan.plannedValues.end());
    const auto &plan = planIt->second;
    auto &encodedValue = _encodedValues[value];

    auto resourceIt = encodedValue.resources.find(resourceKey);
    if (resourceIt == encodedValue.resources.end()) {
        const ResourceRef resourceRef = createResource(plan, resourceKey);
        addSamplerConfig(resourceRef, resourceKey);
        resourceIt = encodedValue.resources.emplace(resourceKey, resourceRef).first;
    }

    return resourceIt->second;
}

BindingSlotRef ResourcePlanEncoder::getOrCreateLogicalBindingSlot(Value value, const ResourceKey &resourceKey) {
    auto planIt = _resourcePlan.plannedValues.find(value);
    assert(planIt != _resourcePlan.plannedValues.end());
    const auto &plan = planIt->second;
    auto &encodedValue = _encodedValues[value];

    auto bindingSlotIt = encodedValue.logicalBindingSlots.find(resourceKey);
    if (bindingSlotIt == encodedValue.logicalBindingSlots.end()) {
        bindingSlotIt = encodedValue.logicalBindingSlots
                            .emplace(resourceKey, _vgfBuilder.getEncoder()->AddBindingSlot(
                                                      plan.bindingIndex, getOrCreateResource(value, resourceKey)))
                            .first;
    }

    return bindingSlotIt->second;
}

BindingSlotRef ResourcePlanEncoder::getOrCreateDescriptorBindingSlot(Value value, const ResourceKey &resourceKey,
                                                                     uint32_t binding) {
    assert(_resourcePlan.plannedValues.find(value) != _resourcePlan.plannedValues.end());
    auto &encodedValue = _encodedValues[value];

    const DescriptorBindingKey key = {resourceKey, binding};
    auto bindingSlotIt = encodedValue.descriptorBindingSlots.find(key);
    if (bindingSlotIt == encodedValue.descriptorBindingSlots.end()) {
        bindingSlotIt =
            encodedValue.descriptorBindingSlots
                .emplace(key,
                         _vgfBuilder.getEncoder()->AddBindingSlot(binding, getOrCreateResource(value, resourceKey)))
                .first;
    }

    return bindingSlotIt->second;
}

void ResourcePlanEncoder::materializeAttachment(const SegmentAttachment &attachment) {
    const auto bindingSlotRef = getOrCreateLogicalBindingSlot(attachment.value, attachment.resourceKey);
    auto &segmentBindings = attachment.isOutput ? _encodedPlan.segmentOutputBindings[attachment.segmentId]
                                                : _encodedPlan.segmentInputBindings[attachment.segmentId];
    segmentBindings.push_back(bindingSlotRef);

    if (attachment.descriptorSet.has_value() && attachment.descriptorBinding.has_value()) {
        addBindingToDescriptorSet(
            attachment.segmentId, *attachment.descriptorSet,
            getOrCreateDescriptorBindingSlot(attachment.value, attachment.resourceKey, *attachment.descriptorBinding));
    }
}

void ResourcePlanEncoder::materializeBindings() {
    for (auto operand : _resourcePlan.sequenceInputValues) {
        auto planIt = _resourcePlan.plannedValues.find(operand);
        if (planIt == _resourcePlan.plannedValues.end()) {
            continue;
        }
        const auto canonicalResource = getCanonicalResource(planIt->second, ResourceCategory::INPUT);
        if (!canonicalResource.has_value()) {
            continue;
        }
        _encodedPlan.sequenceInputBindings.push_back(getOrCreateLogicalBindingSlot(operand, *canonicalResource));
    }

    if (_encodedPlan.sequenceInputBindings.size() < _resourcePlan.sequenceInputValues.size()) {
        llvm::errs() << "Warning: Sequence module contains unused arguments\n";
    }

    for (const auto &attachment : _resourcePlan.sequenceInputAttachments) {
        materializeAttachment(attachment);
    }
    for (const auto &attachment : _resourcePlan.intermediateAttachments) {
        materializeAttachment(attachment);
    }

    for (auto operand : _resourcePlan.sequenceOutputValues) {
        auto planIt = _resourcePlan.plannedValues.find(operand);
        if (planIt == _resourcePlan.plannedValues.end()) {
            continue;
        }
        const auto canonicalResource = getCanonicalResource(planIt->second, ResourceCategory::OUTPUT);
        if (!canonicalResource.has_value()) {
            continue;
        }
        _encodedPlan.sequenceOutputBindings.push_back(getOrCreateLogicalBindingSlot(operand, *canonicalResource));
    }
    for (const auto &attachment : _resourcePlan.sequenceOutputAttachments) {
        materializeAttachment(attachment);
    }
}

} // namespace mlir::model_converter_passes::detail
