/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "conversion/resource_planner.hpp"
#include "include/passes.hpp"
#include "mlir/Target/SPIRV/Serialization.h"
#include "utils.hpp"
#include "vgf/encoder.hpp"
#include "vgf_builder.hpp"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <optional>
#include <vector>

using namespace mlsdk::vgflib;

namespace mlir::model_converter_passes {
#define GEN_PASS_DEF_SERIALIZEVGFPASS
#include "passes.hpp.inc"
namespace {

std::optional<ShaderType> toShaderType(const StringRef language) {
    if (language == "GLSL") {
        return ShaderType::GLSL;
    }
    if (language == "HLSL") {
        return ShaderType::HLSL;
    }
    return std::nullopt;
}

using detail::ResourcePlanEncoder;

class SerializeVGFPass : public impl::SerializeVGFPassBase<SerializeVGFPass> {
  public:
    SerializeVGFPass() : SerializeVGFPass(std::make_shared<VGFBuilder>(), "binary.vgf", {}) {}

    explicit SerializeVGFPass(const SerializeVGFPassOptions &options)
        : SerializeVGFPass(std::make_shared<VGFBuilder>(), "binary.vgf", options) {}

    SerializeVGFPass(std::shared_ptr<VGFBuilder> VGFBuilder, std::string outputName,
                     const SerializeVGFPassOptions &options)
        : _VGFBuilder(std::move(VGFBuilder)), _outputName(std::move(outputName)),
          _emitDebugInfo(options.emitDebugInfo) {}

    void runOnOperation() override {
        if (serialize(getOperation()).failed()) {
            return signalPassFailure();
        }
    }

  private:
    mlir::LogicalResult serializeModule(mlir::vgf::SequenceOp &sequenceOp) {

        // Fetch input/output names if available
        std::vector<std::string> inputNames = {};
        std::vector<std::string> outputNames = {};

        if (auto tfEntryFunctionAttr = sequenceOp->getAttrDictionary().getAs<DictionaryAttr>("tf.entry_function")) {
            if (auto inputsAttr = tfEntryFunctionAttr.getAs<StringAttr>("inputs")) {
                SplitString(inputsAttr.str(), ",", inputNames);
            }
            if (auto outputsAttr = tfEntryFunctionAttr.getAs<StringAttr>("outputs")) {
                SplitString(outputsAttr.str(), ",", outputNames);
            }
        }

        // If there are no names associated to inputs/outputs or the number of names and bindings
        // mismatch, generate unique input/output names
        if (inputNames.empty() || (sequenceOp.getNumArguments() != inputNames.size())) {
            inputNames.clear();
            for (unsigned int inputIdx = 0; inputIdx < sequenceOp.getNumArguments(); ++inputIdx) {
                inputNames.push_back("input_" + std::to_string(inputIdx));
            }
        }
        if (outputNames.empty() || (sequenceOp.getNumResults() != outputNames.size())) {
            outputNames.clear();
            for (unsigned int outputIdx = 0; outputIdx < sequenceOp.getNumResults(); ++outputIdx) {
                outputNames.push_back("output_" + std::to_string(outputIdx));
            }
        }

        detail::ResourcePlanner resourcePlanner(sequenceOp);
        if (resourcePlanner.buildPlan().failed()) {
            return failure();
        }
        ResourcePlanEncoder resourcePlanEncoder(resourcePlanner.getPlan(), *_VGFBuilder);
        const auto &encodedResourcePlan = resourcePlanEncoder.encode();

        uint32_t computeSegmentId = 0;
        uint32_t graphSegmentId = 0;

        WalkResult sequenceWalkResult = sequenceOp.walk([&](vgf::SegmentOp segmentOp) {
            const auto segmentName = segmentOp.getSymName();
            const auto segmentType = segmentOp.getSegmentType();
            auto segmentId = segmentOp->getAttrOfType<IntegerAttr>("segment_id").getUInt();
            const auto &segmentInputBindings = encodedResourcePlan.getSegmentInputBindings(segmentId);
            const auto &segmentOutputBindings = encodedResourcePlan.getSegmentOutputBindings(segmentId);

            WalkResult segmentWalkResult;
            std::vector<BindingSlotRef> segmentAllBindings = {};
            if (segmentType == vgf::SegmentTypeEnum::COMPUTE) {
                segmentWalkResult = segmentOp.walk([&](vgf::ShaderPlaceholderOp shaderPlaceholderOp) {
                    // Add shader module table entry
                    const auto computeModuleRef = [&]() {
                        auto shaderLanguageAttr = shaderPlaceholderOp.getShaderLanguageAttr();
                        if (shaderLanguageAttr) {
                            auto shaderCodeAttr = shaderPlaceholderOp.getShaderCodeAttr();

                            if (shaderLanguageAttr.str() == "SPIR-V") {
                                if (auto shaderBinaryCodeAttr =
                                        llvm::dyn_cast_if_present<DenseI32ArrayAttr>(shaderCodeAttr)) {
                                    std::vector<uint32_t> binaryCode;
                                    binaryCode.reserve(shaderBinaryCodeAttr.asArrayRef().size());
                                    std::transform(shaderBinaryCodeAttr.asArrayRef().begin(),
                                                   shaderBinaryCodeAttr.asArrayRef().end(),
                                                   std::back_inserter(binaryCode),
                                                   [](int32_t word) { return static_cast<uint32_t>(word); });
                                    return _VGFBuilder->getEncoder()->AddModule(
                                        ModuleType::COMPUTE, segmentName.str(),
                                        shaderPlaceholderOp.getEntryPointAttr().str(), binaryCode);
                                }
                            } else if (auto shaderSourceAttr = llvm::dyn_cast_if_present<StringAttr>(shaderCodeAttr)) {
                                const auto shaderType = toShaderType(shaderLanguageAttr.str());
                                if (shaderType.has_value()) {
                                    return _VGFBuilder->getEncoder()->AddModule(
                                        ModuleType::COMPUTE, segmentName.str(),
                                        shaderPlaceholderOp.getEntryPointAttr().str(), shaderType.value(),
                                        shaderSourceAttr.str());
                                }
                            }
                        }

                        return _VGFBuilder->getEncoder()->AddModule(ModuleType::COMPUTE, segmentName.str(),
                                                                    shaderPlaceholderOp.getEntryPointAttr().str());
                    }();

                    const auto descriptorSetInfos = [&]() {
                        std::vector<DescriptorSetInfoRef> descriptorSetInfos = {};
                        auto descriptorSetBindingsIt = encodedResourcePlan.segmentDescriptorSetBindings.find(segmentId);
                        if (descriptorSetBindingsIt != encodedResourcePlan.segmentDescriptorSetBindings.end()) {
                            for (const auto &[descriptorSetIndex, bindings] : descriptorSetBindingsIt->second) {
                                descriptorSetInfos.push_back(
                                    _VGFBuilder->getEncoder()->AddDescriptorSetInfo(bindings, descriptorSetIndex));
                            }
                        }
                        return descriptorSetInfos;
                    }();

                    auto workgroupSizes = ArrayRef<int64_t>(shaderPlaceholderOp.getWorkgroupSizesAttr());
                    assert(workgroupSizes.size() == 3);
                    const std::array<uint32_t, 3> dispatchShape = {static_cast<uint32_t>(workgroupSizes[0]),
                                                                   static_cast<uint32_t>(workgroupSizes[1]),
                                                                   static_cast<uint32_t>(workgroupSizes[2])};

                    _VGFBuilder->getEncoder()->AddSegmentInfo(
                        computeModuleRef, "compute_segment_" + std::to_string(computeSegmentId++), descriptorSetInfos,
                        segmentInputBindings, segmentOutputBindings, _VGFBuilder->getConstantRefs(), dispatchShape);

                    return WalkResult::advance();
                });
            } else if (segmentType == vgf::SegmentTypeEnum::GRAPH) {
                segmentWalkResult = segmentOp.walk([&](spirv::ModuleOp spirvModuleOp) {
                    mlir::spirv::SerializationOptions options;
                    options.emitDebugInfo = _emitDebugInfo;
                    SmallVector<uint32_t> binary;
                    if (spirv::serialize(spirvModuleOp, binary, options).failed()) {
                        return WalkResult::interrupt();
                    }

                    std::string entryPointName;
                    spirvModuleOp.walk([&](spirv::GraphEntryPointARMOp opGraphEntryPoint) {
                        entryPointName = opGraphEntryPoint.getFn().str();
                    });

                    // Add graph module table entry
                    ModuleRef graphModuleRef =
                        _VGFBuilder->getEncoder()->AddModule(ModuleType::GRAPH, segmentName.str(), entryPointName,
                                                             std::vector<uint32_t>(binary.begin(), binary.end()));

                    WalkResult spirvModuleWalkResult = spirvModuleOp.walk([&](spirv::GraphARMOp) {
                        std::copy(segmentInputBindings.cbegin(), segmentInputBindings.cend(),
                                  std::back_inserter(segmentAllBindings));
                        std::copy(segmentOutputBindings.cbegin(), segmentOutputBindings.cend(),
                                  std::back_inserter(segmentAllBindings));

                        const DescriptorSetInfoRef descSetInfo =
                            _VGFBuilder->getEncoder()->AddDescriptorSetInfo(segmentAllBindings);

                        _VGFBuilder->getEncoder()->AddSegmentInfo(
                            graphModuleRef, "graph_segment_" + std::to_string(graphSegmentId++), {descSetInfo},
                            segmentInputBindings, segmentOutputBindings, _VGFBuilder->getConstantRefs());

                        return WalkResult::advance();
                    });
                    return spirvModuleWalkResult;
                });
            } else {
                // Segments can either be of Graph or Compute types
                return WalkResult::interrupt();
            }
            return segmentWalkResult;
        });

        if (sequenceWalkResult.wasInterrupted()) {
            return failure();
        }

        _VGFBuilder->getEncoder()->AddModelSequenceInputsOutputs(encodedResourcePlan.sequenceInputBindings, inputNames,
                                                                 encodedResourcePlan.sequenceOutputBindings,
                                                                 outputNames);
        return success();
    }

    LogicalResult serialize(vgf::SequenceOp sequenceOp) {
        if (serializeModule(sequenceOp).failed()) {
            llvm::errs() << "Unable to serialize VGF module\n";
            return failure();
        }

        _VGFBuilder->getEncoder()->Finish();

        if (_outputName == "-") {
            if (!_VGFBuilder->getEncoder()->WriteTo(std::cout)) {
                llvm::errs() << "Unable to write to stdout\n";
                return failure();
            }
            return success();
        }

        std::ofstream fstream(_outputName, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!fstream.is_open()) {
            llvm::errs() << "Unable to create output file\n";
            return failure();
        }

        if (!_VGFBuilder->getEncoder()->WriteTo(fstream)) {
            llvm::errs() << "Error writing to file\n";
            return failure();
        }

        llvm::outs() << "Successfully saved vgf output to \"" << _outputName << "\"\n";

        return success();
    }

    std::shared_ptr<VGFBuilder> _VGFBuilder = {nullptr};
    std::string _outputName;
    bool _emitDebugInfo;
};

} // namespace

std::unique_ptr<Pass> createSerializeVGFPass(std::shared_ptr<VGFBuilder> VGFBuilder, std::string outputName,
                                             const SerializeVGFPassOptions &options) {
    return std::make_unique<SerializeVGFPass>(VGFBuilder, outputName, options);
}

} // namespace mlir::model_converter_passes
