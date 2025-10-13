/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "include/passes.hpp"
#include "mlir/Target/SPIRV/Serialization.h"
#include "utils.hpp"
#include "vgf/encoder.hpp"
#include "vgf_builder.hpp"

#define VGFLIB_VK_HELPERS
#include "vgf/vulkan_helpers.generated.hpp"

#include <fstream>
#include <iostream>

using namespace mlsdk::vgflib;

namespace mlir {
namespace model_converter_passes {
namespace {

using SegmentId = uint64_t;

// As defined in vulkan_core.h
// FIXME: We may choose to link in vulkan_headers directly once we need to
// support more types, but for now we just need this one value so here it is.
static constexpr DescriptorType DESCRIPTOR_TYPE_TENSOR_ARM = 1000460000;

void setGlobalVarOpBindingAndDescriptorSet(spirv::GraphARMOp opGraph, Value operand, uint32_t bindingId) {
    auto runSegmentOp = opGraph->getParentOp()->getParentOp()->getNextNode();
    if (auto moduleOp = llvm::dyn_cast<spirv::ModuleOp>(opGraph->getParentOp())) {

        SmallVector<Value> rangeOperandsAndResults = ValueRange(runSegmentOp->getOperands());
        rangeOperandsAndResults.append(runSegmentOp->getResults().begin(), runSegmentOp->getResults().end());

        auto it = std::find(rangeOperandsAndResults.begin(), rangeOperandsAndResults.end(), operand);
        if (it == rangeOperandsAndResults.end()) {
            return;
        }
        auto operandInterfaceId = std::distance(rangeOperandsAndResults.begin(), it);

        auto graphEntryPointOp = *moduleOp.getBody()->getOps<spirv::GraphEntryPointARMOp>().begin();
        for (auto globalVarOp : moduleOp.getBody()->getOps<spirv::GlobalVariableOp>()) {
            if (globalVarOp.getNameAttr().getValue() ==
                llvm::cast<FlatSymbolRefAttr>(
                    graphEntryPointOp.getInterface()[static_cast<unsigned>(operandInterfaceId)])
                    .getValue()) {
                globalVarOp.setBinding(bindingId);
                // TODO: DescriptorSet values to be handled correctly, currently they are set to 0
                globalVarOp.setDescriptorSet(0);
                return;
            }
        }
    }
}

class SerializeVGFPass : public SerializeVGFPassBase<SerializeVGFPass> {
  public:
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

        auto isSequenceInputOperand = [&](Value operand) {
            return std::any_of(sequenceOp.getArguments().begin(), sequenceOp.getArguments().end(),
                               [&](const Value sequenceOpArgument) { return sequenceOpArgument == operand; });
        };

        auto sequenceOutputOp = sequenceOp.front().getTerminator();
        auto isSequenceOutputOperand = [&](Value operand) {
            return std::any_of(sequenceOutputOp->getOperands().begin(), sequenceOutputOp->getOperands().end(),
                               [&](const Value sequenceOutputOperand) { return sequenceOutputOperand == operand; });
        };

        DenseMap<Value, std::shared_ptr<BindingSlotRef>> mapOperandsAndBindingRef;
        std::map<SegmentId, std::vector<BindingSlotRef>> mapSegmentInputBindings = {};
        std::map<SegmentId, std::vector<BindingSlotRef>> mapSegmentOutputBindings = {};

        // First: Resolve sequence input resources' bindings
        std::vector<BindingSlotRef> sequenceInputBindings = {};
        for (auto operand : sequenceOp.getArguments()) {
            sequenceOp.walk([&](vgf::SegmentOp segmentOp) {
                const auto segmentType = segmentOp.getSegmentType();
                auto segmentId = segmentOp->getAttrOfType<IntegerAttr>("segment_id").getUInt();
                auto runSegmentOp = segmentOp->getNextNode();

                WalkResult segmentWalkResult;
                if (segmentType == vgf::SegmentTypeEnum::COMPUTE) {
                    segmentWalkResult = segmentOp.walk([&](vgf::ShaderPlaceholderOp shaderPlaceholderOp) {
                        for (const auto &[inputDescriptorType, inputVkFormat, input] :
                             llvm::zip(shaderPlaceholderOp.getInputVkDescriptorTypes(),
                                       shaderPlaceholderOp.getInputVkFormats(), runSegmentOp->getOperands())) {
                            if (input == operand &&
                                mapOperandsAndBindingRef.find(input) == mapOperandsAndBindingRef.end()) {
                                const ShapedType type = llvm::dyn_cast<ShapedType>(input.getType());
                                auto resourceRef =
                                    std::make_shared<ResourceRef>(_VGFBuilder->getEncoder()->AddInputResource(
                                        NameToDescriptorType(llvm::cast<StringAttr>(inputDescriptorType).str()),
                                        NameToFormatType(llvm::cast<StringAttr>(inputVkFormat).str()), type.getShape(),
                                        {}));

                                auto bindingSlotRef = std::make_shared<BindingSlotRef>(
                                    _VGFBuilder->getEncoder()->AddBindingSlot(operand.getArgNumber(), *resourceRef));

                                mapOperandsAndBindingRef[input] = bindingSlotRef;
                                sequenceInputBindings.push_back(*bindingSlotRef);
                                mapSegmentInputBindings[segmentId].push_back(*bindingSlotRef);
                            }
                        }

                        return WalkResult::advance();
                    });
                } else if (segmentType == vgf::SegmentTypeEnum::GRAPH) {
                    segmentWalkResult = segmentOp.walk([&](spirv::ModuleOp spirvModuleOp) {
                        WalkResult spirvModuleWalkResult = spirvModuleOp.walk([&](spirv::GraphARMOp) {
                            for (const auto &[inputIdx, input] : llvm::enumerate(runSegmentOp->getOperands())) {
                                if (input == operand &&
                                    mapOperandsAndBindingRef.find(input) == mapOperandsAndBindingRef.end()) {
                                    VGFBuilder::VkFormat vkFormat;
                                    const ShapedType type = llvm::dyn_cast<ShapedType>(input.getType());

                                    if (_VGFBuilder
                                            ->mlirTypeToVkFormat(
                                                type.getElementType(), vkFormat,
                                                sequenceOp
                                                    .getArgAttrOfType<BoolAttr>(static_cast<uint32_t>(inputIdx),
                                                                                "mlsdk.unsigned_input_output")
                                                    .getValue())
                                            .failed()) {
                                        llvm::errs() << "Unsupported type for input at index " << inputIdx
                                                     << " in segment " << segmentOp.getSymName().str() << "\n";
                                        return WalkResult::interrupt();
                                    }

                                    auto resourceRef = _VGFBuilder->getEncoder()->AddInputResource(
                                        DESCRIPTOR_TYPE_TENSOR_ARM, static_cast<FormatType>(vkFormat), type.getShape(),
                                        {});

                                    auto bindingSlotRef = std::make_shared<BindingSlotRef>(
                                        _VGFBuilder->getEncoder()->AddBindingSlot(operand.getArgNumber(), resourceRef));

                                    mapOperandsAndBindingRef[input] = bindingSlotRef;
                                    sequenceInputBindings.push_back(*bindingSlotRef);
                                    mapSegmentInputBindings[segmentId].push_back(*bindingSlotRef);
                                }
                            }

                            return WalkResult::advance();
                        });
                        return spirvModuleWalkResult;
                    });
                } else {
                    // Segments can either be of Graph or Compute types
                    llvm::errs() << "Invalid segment type in sequence module\n";
                    return WalkResult::interrupt();
                }
                return segmentWalkResult;
            });
        }

        if (sequenceInputBindings.size() < sequenceOp.getNumArguments()) {
            llvm::errs() << "Warning: Sequence module contains unused arguments\n";
        }

        // Second: Resolve sequence intermediate resources' bindings
        auto bindingId = sequenceOp.getNumArguments();
        WalkResult sequenceWalkResult = sequenceOp.walk([&](vgf::SegmentOp segmentOp) {
            const auto segmentType = segmentOp.getSegmentType();
            auto segmentId = segmentOp->getAttrOfType<IntegerAttr>("segment_id").getUInt();
            auto runSegmentOp = segmentOp->getNextNode();

            WalkResult segmentWalkResult;
            std::vector<BindingSlotRef> segmentInputBindings = {};
            if (segmentType == vgf::SegmentTypeEnum::COMPUTE) {
                segmentWalkResult = segmentOp.walk([&](vgf::ShaderPlaceholderOp shaderPlaceholderOp) {
                    for (const auto &[inputDescriptorType, inputVkFormat, input] :
                         llvm::zip(shaderPlaceholderOp.getInputVkDescriptorTypes(),
                                   shaderPlaceholderOp.getInputVkFormats(), runSegmentOp->getOperands())) {
                        if (!isSequenceInputOperand(input)) {
                            auto it = mapOperandsAndBindingRef.find(input);
                            if (it == mapOperandsAndBindingRef.end()) {
                                // TODO: Revisit the need for conversion to ShapedType
                                const ShapedType type = convertShapedType(input.getType());

                                auto resourceRef =
                                    std::make_shared<ResourceRef>(_VGFBuilder->getEncoder()->AddIntermediateResource(
                                        NameToDescriptorType(llvm::cast<StringAttr>(inputDescriptorType).str()),
                                        NameToFormatType(llvm::cast<StringAttr>(inputVkFormat).str()), type.getShape(),
                                        {}));

                                auto bindingSlotRef = std::make_shared<BindingSlotRef>(
                                    _VGFBuilder->getEncoder()->AddBindingSlot(bindingId++, *resourceRef));

                                mapOperandsAndBindingRef[input] = bindingSlotRef;
                                mapSegmentInputBindings[segmentId].push_back(*bindingSlotRef);
                            } else {
                                mapSegmentInputBindings[segmentId].push_back(*(it->second));
                            }
                        }
                    }

                    for (const auto &[outputDescriptorType, outputVkFormat, result] :
                         llvm::zip(shaderPlaceholderOp.getOutputVkDescriptorTypes(),
                                   shaderPlaceholderOp.getOutputVkFormats(), runSegmentOp->getResults())) {
                        if (!isSequenceOutputOperand(result)) {
                            auto it = mapOperandsAndBindingRef.find(result);
                            if (it == mapOperandsAndBindingRef.end()) {
                                // TODO: Revisit the need for conversion to ShapedType
                                const ShapedType type = convertShapedType(result.getType());

                                auto resourceRef =
                                    std::make_shared<ResourceRef>(_VGFBuilder->getEncoder()->AddIntermediateResource(
                                        NameToDescriptorType(llvm::cast<StringAttr>(outputDescriptorType).str()),
                                        NameToFormatType(llvm::cast<StringAttr>(outputVkFormat).str()), type.getShape(),
                                        {}));

                                auto bindingSlotRef = std::make_shared<BindingSlotRef>(
                                    _VGFBuilder->getEncoder()->AddBindingSlot(bindingId++, *resourceRef));

                                mapOperandsAndBindingRef[result] = bindingSlotRef;
                                mapSegmentOutputBindings[segmentId].push_back(*bindingSlotRef);
                            } else {
                                mapSegmentOutputBindings[segmentId].push_back(*(it->second));
                            }
                        }
                    }
                    return WalkResult::advance();
                });
            } else if (segmentType == vgf::SegmentTypeEnum::GRAPH) {
                segmentWalkResult = segmentOp.walk([&](spirv::ModuleOp spirvModuleOp) {
                    WalkResult spirvModuleWalkResult = spirvModuleOp.walk([&](spirv::GraphARMOp) {
                        for (const auto &input : runSegmentOp->getOperands()) {
                            if (!isSequenceInputOperand(input)) {
                                auto it = mapOperandsAndBindingRef.find(input);
                                if (it == mapOperandsAndBindingRef.end()) {
                                    VGFBuilder::VkFormat vkFormat;
                                    // TODO: Revisit the need for conversion to ShapedType
                                    const ShapedType type = convertShapedType(input.getType());
                                    if (_VGFBuilder->mlirTypeToVkFormat(type.getElementType(), vkFormat, false)
                                            .failed()) {
                                        llvm::errs() << "Unsupported type for input in segment "
                                                     << segmentOp.getSymName().str() << "\n";
                                        return WalkResult::interrupt();
                                    }

                                    auto resourceRef = _VGFBuilder->getEncoder()->AddIntermediateResource(
                                        DESCRIPTOR_TYPE_TENSOR_ARM, static_cast<FormatType>(vkFormat), type.getShape(),
                                        {});

                                    auto bindingSlotRef = std::make_shared<BindingSlotRef>(
                                        _VGFBuilder->getEncoder()->AddBindingSlot(bindingId++, resourceRef));

                                    mapOperandsAndBindingRef[input] = bindingSlotRef;
                                    mapSegmentInputBindings[segmentId].push_back(*bindingSlotRef);
                                } else {
                                    mapSegmentInputBindings[segmentId].push_back(*(it->second));
                                }
                            }
                        }

                        for (const auto &result : runSegmentOp->getResults()) {
                            if (!isSequenceOutputOperand(result)) {
                                auto it = mapOperandsAndBindingRef.find(result);
                                if (it == mapOperandsAndBindingRef.end()) {
                                    VGFBuilder::VkFormat vkFormat;
                                    // TODO: Revisit the need for conversion to ShapedType
                                    const ShapedType type = convertShapedType(result.getType());
                                    if (_VGFBuilder->mlirTypeToVkFormat(type.getElementType(), vkFormat, false)
                                            .failed()) {
                                        llvm::errs() << "Unsupported type for result in segment "
                                                     << segmentOp.getSymName().str() << "\n";
                                        return WalkResult::interrupt();
                                    }

                                    auto resourceRef = std::make_shared<ResourceRef>(
                                        _VGFBuilder->getEncoder()->AddIntermediateResource(
                                            DESCRIPTOR_TYPE_TENSOR_ARM, static_cast<FormatType>(vkFormat),
                                            type.getShape(), {}));

                                    auto bindingSlotRef = std::make_shared<BindingSlotRef>(
                                        _VGFBuilder->getEncoder()->AddBindingSlot(bindingId++, *resourceRef));

                                    mapOperandsAndBindingRef[result] = bindingSlotRef;
                                    mapSegmentOutputBindings[segmentId].push_back(*bindingSlotRef);
                                } else {
                                    mapSegmentOutputBindings[segmentId].push_back(*(it->second));
                                }
                            }
                        }
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

        // Third: Resolve sequence output resources' bindings
        std::vector<BindingSlotRef> sequenceOutputBindings = {};
        for (auto operand : sequenceOutputOp->getOperands()) {
            sequenceOp.walk([&](vgf::SegmentOp segmentOp) {
                const auto segmentType = segmentOp.getSegmentType();
                auto segmentId = segmentOp->getAttrOfType<IntegerAttr>("segment_id").getUInt();
                auto runSegmentOp = segmentOp->getNextNode();

                WalkResult segmentWalkResult;
                if (segmentType == vgf::SegmentTypeEnum::COMPUTE) {
                    segmentWalkResult = segmentOp.walk([&](vgf::ShaderPlaceholderOp shaderPlaceholderOp) {
                        for (const auto &[outputDescriptorType, outputVkFormat, result] :
                             llvm::zip(shaderPlaceholderOp.getOutputVkDescriptorTypes(),
                                       shaderPlaceholderOp.getOutputVkFormats(), runSegmentOp->getResults())) {

                            if (result == operand &&
                                mapOperandsAndBindingRef.find(result) == mapOperandsAndBindingRef.end()) {
                                const ShapedType type = llvm::dyn_cast<ShapedType>(result.getType());

                                auto resourceRef =
                                    std::make_shared<ResourceRef>(_VGFBuilder->getEncoder()->AddOutputResource(
                                        NameToDescriptorType(llvm::cast<StringAttr>(outputDescriptorType).str()),
                                        NameToFormatType(llvm::cast<StringAttr>(outputVkFormat).str()), type.getShape(),
                                        {}));

                                auto bindingSlotRef = std::make_shared<BindingSlotRef>(
                                    _VGFBuilder->getEncoder()->AddBindingSlot(bindingId++, *resourceRef));

                                mapOperandsAndBindingRef[result] = bindingSlotRef;
                                sequenceOutputBindings.push_back(*bindingSlotRef);
                                mapSegmentOutputBindings[segmentId].push_back(*bindingSlotRef);
                            }
                        }

                        return WalkResult::advance();
                    });
                } else if (segmentType == vgf::SegmentTypeEnum::GRAPH) {
                    segmentWalkResult = segmentOp.walk([&](spirv::ModuleOp spirvModuleOp) {
                        WalkResult spirvModuleWalkResult = spirvModuleOp.walk([&](spirv::GraphARMOp) {
                            for (const auto &[resIdx, result] : llvm::enumerate(runSegmentOp->getResults())) {
                                if (result == operand &&
                                    mapOperandsAndBindingRef.find(result) == mapOperandsAndBindingRef.end()) {
                                    VGFBuilder::VkFormat vkFormat;
                                    const ShapedType type = llvm::dyn_cast<ShapedType>(result.getType());

                                    if (_VGFBuilder
                                            ->mlirTypeToVkFormat(
                                                type.getElementType(), vkFormat,
                                                sequenceOp
                                                    .getResultAttrOfType<BoolAttr>(static_cast<uint32_t>(resIdx),
                                                                                   "mlsdk.unsigned_input_output")
                                                    .getValue())
                                            .failed()) {
                                        llvm::errs() << "Unsupported type for result at index " << resIdx
                                                     << " in segment " << segmentOp.getSymName().str() << "\n";
                                        return WalkResult::interrupt();
                                    }

                                    auto resourceRef = _VGFBuilder->getEncoder()->AddOutputResource(
                                        DESCRIPTOR_TYPE_TENSOR_ARM, static_cast<FormatType>(vkFormat), type.getShape(),
                                        {});

                                    auto bindingSlotRef = std::make_shared<BindingSlotRef>(
                                        _VGFBuilder->getEncoder()->AddBindingSlot(bindingId++, resourceRef));

                                    mapOperandsAndBindingRef[result] = bindingSlotRef;
                                    sequenceOutputBindings.push_back(*bindingSlotRef);
                                    mapSegmentOutputBindings[segmentId].push_back(*bindingSlotRef);
                                }
                            }

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
        }

        // Resolve Bindings and DescriptorSets in SPIR-V Modules
        // TODO: This can perhaps be done in a better way
        sequenceOp.walk([&](vgf::SegmentOp segmentOp) {
            if (segmentOp.getSegmentType() != vgf::SegmentTypeEnum::GRAPH) {
                return;
            }
            segmentOp.walk([&](spirv::GraphARMOp opGraph) {
                for (const auto &[operand, bindingSlotRef] : mapOperandsAndBindingRef) {
                    setGlobalVarOpBindingAndDescriptorSet(opGraph, operand, bindingSlotRef->reference);
                }
            });
        });

        uint32_t computeSegmentId = 0;
        uint32_t graphSegmentId = 0;

        sequenceWalkResult = sequenceOp.walk([this, &computeSegmentId, &graphSegmentId, &mapSegmentInputBindings,
                                              &mapSegmentOutputBindings](vgf::SegmentOp segmentOp) {
            const auto segmentName = segmentOp.getSymName();
            const auto segmentType = segmentOp.getSegmentType();
            auto segmentId = segmentOp->getAttrOfType<IntegerAttr>("segment_id").getUInt();

            WalkResult segmentWalkResult;
            std::vector<BindingSlotRef> segmentAllBindings = {};
            if (segmentType == vgf::SegmentTypeEnum::COMPUTE) {
                segmentWalkResult = segmentOp.walk([&](vgf::ShaderPlaceholderOp shaderPlaceholderOp) {
                    // Add shader module table entry
                    ModuleRef computeModuleRef = _VGFBuilder->getEncoder()->AddModule(
                        ModuleType::COMPUTE, segmentName.str(), shaderPlaceholderOp.getEntryPointAttr().str());

                    std::copy(mapSegmentInputBindings[segmentId].cbegin(), mapSegmentInputBindings[segmentId].cend(),
                              std::back_inserter(segmentAllBindings));
                    std::copy(mapSegmentOutputBindings[segmentId].cbegin(), mapSegmentOutputBindings[segmentId].cend(),
                              std::back_inserter(segmentAllBindings));

                    const DescriptorSetInfoRef descSetInfo =
                        _VGFBuilder->getEncoder()->AddDescriptorSetInfo(segmentAllBindings);

                    auto workgroupSizes = ArrayRef<int64_t>(shaderPlaceholderOp.getWorkgroupSizesAttr());
                    assert(workgroupSizes.size() == 3);
                    const std::array<uint32_t, 3> dispatchShape = {static_cast<uint32_t>(workgroupSizes[0]),
                                                                   static_cast<uint32_t>(workgroupSizes[1]),
                                                                   static_cast<uint32_t>(workgroupSizes[2])};

                    _VGFBuilder->getEncoder()->AddSegmentInfo(
                        computeModuleRef, "compute_segment_" + std::to_string(computeSegmentId++), {descSetInfo},
                        mapSegmentInputBindings[segmentId], mapSegmentOutputBindings[segmentId],
                        _VGFBuilder->getConstantRefs(), dispatchShape);

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
                        std::copy(mapSegmentInputBindings[segmentId].cbegin(),
                                  mapSegmentInputBindings[segmentId].cend(), std::back_inserter(segmentAllBindings));
                        std::copy(mapSegmentOutputBindings[segmentId].cbegin(),
                                  mapSegmentOutputBindings[segmentId].cend(), std::back_inserter(segmentAllBindings));

                        const DescriptorSetInfoRef descSetInfo =
                            _VGFBuilder->getEncoder()->AddDescriptorSetInfo(segmentAllBindings);

                        _VGFBuilder->getEncoder()->AddSegmentInfo(
                            graphModuleRef, "graph_segment_" + std::to_string(graphSegmentId++), {descSetInfo},
                            mapSegmentInputBindings[segmentId], mapSegmentOutputBindings[segmentId],
                            _VGFBuilder->getConstantRefs());

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

        _VGFBuilder->getEncoder()->AddModelSequenceInputsOutputs(sequenceInputBindings, inputNames,
                                                                 sequenceOutputBindings, outputNames);
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

void registerSerializeVGFPass() {
    PassRegistration<SerializeVGFPass>([]() -> std::unique_ptr<Pass> {
        return createSerializeVGFPass(std::make_shared<VGFBuilder>(), "binary.vgf", {false});
    });
}

} // namespace model_converter_passes
} // namespace mlir
