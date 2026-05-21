/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "include/passes.hpp"
#include "vgf/encoder.hpp"
#include "vgf_builder.hpp"

#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

namespace mlir::model_converter_passes::detail {

using namespace mlsdk::vgflib;

using SegmentId = uint64_t;

constexpr uint32_t UNSET_SAMPLER_VALUE = std::numeric_limits<uint32_t>::max();

struct SamplerConfigValues {
    uint32_t minFilter = UNSET_SAMPLER_VALUE;
    uint32_t magFilter = UNSET_SAMPLER_VALUE;
    uint32_t addressModeU = UNSET_SAMPLER_VALUE;
    uint32_t addressModeV = UNSET_SAMPLER_VALUE;
    uint32_t borderColor = UNSET_SAMPLER_VALUE;

    bool operator==(const SamplerConfigValues &other) const {
        return minFilter == other.minFilter && magFilter == other.magFilter && addressModeU == other.addressModeU &&
               addressModeV == other.addressModeV && borderColor == other.borderColor;
    }

    bool operator<(const SamplerConfigValues &other) const {
        return std::tie(minFilter, magFilter, addressModeU, addressModeV, borderColor) <
               std::tie(other.minFilter, other.magFilter, other.addressModeU, other.addressModeV, other.borderColor);
    }
};

struct ResourceViewKey {
    DescriptorType descriptorType;
    FormatType vkFormat;

    bool operator==(const ResourceViewKey &other) const {
        return descriptorType == other.descriptorType && vkFormat == other.vkFormat;
    }

    bool operator<(const ResourceViewKey &other) const {
        return std::tie(descriptorType, vkFormat) < std::tie(other.descriptorType, other.vkFormat);
    }
};

struct ResourceKey {
    ResourceCategory category;
    ResourceViewKey view;
    std::optional<SamplerConfigValues> samplerConfig;

    bool operator==(const ResourceKey &other) const {
        return category == other.category && view == other.view && samplerConfig == other.samplerConfig;
    }

    bool operator<(const ResourceKey &other) const {
        return std::tie(category, view.descriptorType, view.vkFormat, samplerConfig) <
               std::tie(other.category, other.view.descriptorType, other.view.vkFormat, other.samplerConfig);
    }
};

struct DescriptorBindingKey {
    ResourceKey resourceKey;
    uint32_t binding;

    bool operator<(const DescriptorBindingKey &other) const {
        return std::tie(resourceKey, binding) < std::tie(other.resourceKey, other.binding);
    }
};

struct PlannedValue {
    uint32_t bindingIndex = 0;
    bool bindingAssigned = false;
    std::vector<int64_t> shape;
    std::vector<ResourceKey> resourceOrder;
    std::optional<ResourceKey> producerResource;
    std::optional<AliasGroupId> aliasGroupId;
};

struct SegmentAttachment {
    SegmentId segmentId;
    Value value;
    bool isOutput = false;
    ResourceKey resourceKey;
    std::optional<int64_t> descriptorSet;
    std::optional<uint32_t> descriptorBinding;
};

struct ResourcePlan {
    DenseMap<Value, PlannedValue> plannedValues;
    std::vector<Value> sequenceInputValues;
    std::vector<Value> intermediateValues;
    std::vector<Value> sequenceOutputValues;
    std::vector<SegmentAttachment> sequenceInputAttachments;
    std::vector<SegmentAttachment> intermediateAttachments;
    std::vector<SegmentAttachment> sequenceOutputAttachments;
};

struct EncodedResourcePlan {
    std::vector<BindingSlotRef> sequenceInputBindings;
    std::vector<BindingSlotRef> sequenceOutputBindings;
    std::map<SegmentId, std::vector<BindingSlotRef>> segmentInputBindings;
    std::map<SegmentId, std::vector<BindingSlotRef>> segmentOutputBindings;
    std::map<SegmentId, std::map<uint32_t, std::vector<BindingSlotRef>>> segmentDescriptorSetBindings;

    const std::vector<BindingSlotRef> &getSegmentInputBindings(SegmentId segmentId) const;
    const std::vector<BindingSlotRef> &getSegmentOutputBindings(SegmentId segmentId) const;
};

struct EncodedPlannedValue {
    std::map<ResourceKey, ResourceRef> resources;
    std::map<ResourceKey, BindingSlotRef> logicalBindingSlots;
    std::map<DescriptorBindingKey, BindingSlotRef> descriptorBindingSlots;
};

class ResourcePlanner {
  public:
    explicit ResourcePlanner(vgf::SequenceOp &sequenceOp);

    LogicalResult buildPlan();

    const ResourcePlan &getPlan() const;

  private:
    bool isSequenceInputOperand(Value operand) const;
    bool isSequenceOutputOperand(Value operand) const;
    bool getSequenceInputUnsigned(uint32_t inputIndex) const;
    bool getSequenceOutputUnsigned(uint32_t outputIndex) const;
    FailureOr<std::vector<int64_t>> getValueShape(Value value) const;
    LogicalResult getGraphView(Value value, StringRef role, StringRef segmentName, ResourceViewKey &view) const;
    static ResourceKey makeResourceKey(ResourceCategory category, const ResourceViewKey &view,
                                       std::optional<SamplerConfigValues> samplerConfig = std::nullopt);
    FailureOr<PlannedValue *> ensurePlannedValue(Value value, std::optional<uint32_t> bindingIndex = std::nullopt);
    void appendResourceRequirement(PlannedValue &plan, const ResourceKey &resourceKey, bool isProducer);
    void appendAttachment(std::vector<SegmentAttachment> &attachments, SegmentId segmentId, Value value, bool isOutput,
                          const ResourceKey &resourceKey, std::optional<int64_t> descriptorSet = std::nullopt,
                          std::optional<uint32_t> descriptorBinding = std::nullopt, bool isProducer = false);
    LogicalResult collectSequenceInputs();
    LogicalResult collectIntermediates();
    bool hasResourceCategory(const PlannedValue &plan, ResourceCategory category) const;
    std::optional<ResourceKey> getCanonicalResource(const PlannedValue &plan, ResourceCategory category) const;
    LogicalResult collectSequenceOutputs();
    void finalizePlannedValues();
    void finalizePlannedValue(Value value);

    vgf::SequenceOp &_sequenceOp;
    Operation *_sequenceOutputOp = nullptr;
    DenseMap<Value, uint32_t> _sequenceOutputIndices;
    ResourcePlan _resourcePlan;
    AliasGroupId _nextAliasGroupId = 0;
    uint32_t _bindingId = 0;
};

class ResourcePlanEncoder {
  public:
    ResourcePlanEncoder(const ResourcePlan &resourcePlan, VGFBuilder &vgfBuilder);

    const EncodedResourcePlan &encode();

  private:
    static std::optional<ResourceKey> getCanonicalResource(const PlannedValue &plan, ResourceCategory category);
    void addBindingToDescriptorSet(SegmentId segmentId, int64_t descriptorSet, BindingSlotRef bindingSlotRef);
    ResourceRef createResource(const PlannedValue &plan, const ResourceKey &resourceKey);
    void addSamplerConfig(ResourceRef resourceRef, const ResourceKey &resourceKey);
    ResourceRef getOrCreateResource(Value value, const ResourceKey &resourceKey);
    BindingSlotRef getOrCreateLogicalBindingSlot(Value value, const ResourceKey &resourceKey);
    BindingSlotRef getOrCreateDescriptorBindingSlot(Value value, const ResourceKey &resourceKey, uint32_t binding);
    void materializeAttachment(const SegmentAttachment &attachment);
    void materializeBindings();

    const ResourcePlan &_resourcePlan;
    VGFBuilder &_vgfBuilder;
    EncodedResourcePlan _encodedPlan;
    DenseMap<Value, EncodedPlannedValue> _encodedValues;
    std::map<SegmentId, std::map<uint32_t, std::set<uint32_t>>> _segmentDescriptorSetBindingRefs;
};

} // namespace mlir::model_converter_passes::detail
