/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "include/passes.hpp"
#include "mlir/Conversion/TosaToSPIRVTosa/TosaToSPIRVTosa.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "utils.hpp"
#include "vgf_builder.hpp"
#include "llvm/ADT/STLExtras.h"

#ifdef BLOCK_SCALED_TENSOR
#    include "tosa_serialization_handler.h"
#endif

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <vector>

using namespace mlsdk::vgflib;

namespace mlir::model_converter_passes {
#define GEN_PASS_DEF_VGFCONSTANTSPASS
#include "passes.hpp.inc"
namespace {

#ifdef BLOCK_SCALED_TENSOR
// TODO: Move block-scaled constant extraction into the VGF constants pass in LLVM/MLIR.
struct BlockScaledConstant {
    mlir::Type valueType;
    mlir::Type scaleType;
    // Shape metadata is consumed by the serialization follow-up.
    // cppcheck-suppress unusedStructMember
    std::vector<int64_t> shape;
    // cppcheck-suppress unusedStructMember
    std::vector<uint32_t> blockShape;
    std::vector<uint8_t> valueBytes;
    std::vector<uint8_t> scaleBytes;
};

template <typename T> std::vector<T> decodeBits(const std::vector<uint8_t> &bits) {
    auto values =
        llvm::map_range(bits, [](uint8_t value) { return T::from_bits(static_cast<typename T::storage_t>(value)); });
    return {values.begin(), values.end()};
}

mlir::LogicalResult packValueBytes(mlir::tosa::ConstOp constOp, mlir::Type valueType, std::vector<uint8_t> &bytes) {
    const bool isFp4 = mlir::isa<mlir::Float4E2M1FNType>(valueType);
    const bool isFp6 = mlir::isa<mlir::Float6E3M2FNType, mlir::Float6E2M3FNType>(valueType);
    if (!isFp4 && !isFp6) {
        return mlir::success();
    }
    const uint8_t maximumEncoding = isFp4 ? 0x0f : 0x3f;
    if (std::any_of(bytes.begin(), bytes.end(), [maximumEncoding](uint8_t value) { return value > maximumEncoding; })) {
        return constOp.emitError(isFp4 ? "FP4 value encoding exceeds four bits"
                                       : "FP6 value encoding exceeds six bits");
    }

    std::vector<uint8_t> packedBytes;
    ::tosa::tosa_err_t packingResult = ::tosa::TOSA_OK;
    if (isFp4) {
        packingResult =
            ::tosa::TosaSerializationHandler::ConvertFP4E2M1toU8(decodeBits<ct::fp4_e2m1>(bytes), packedBytes);
    } else if (mlir::isa<mlir::Float6E2M3FNType>(valueType)) {
        packingResult =
            ::tosa::TosaSerializationHandler::ConvertFP6E2M3toU8(decodeBits<ct::fp6_e2m3>(bytes), packedBytes);
    } else {
        packingResult =
            ::tosa::TosaSerializationHandler::ConvertFP6E3M2toU8(decodeBits<ct::fp6_e3m2>(bytes), packedBytes);
    }

    if (packingResult != ::tosa::TOSA_OK) {
        return constOp.emitError("failed to apply TOSA host packing to block-scaled values");
    }
    bytes = std::move(packedBytes);
    return mlir::success();
}

/// Extract packed values and E8M0 scales, reporting errors on the constant.
[[maybe_unused]] mlir::FailureOr<BlockScaledConstant> extractBlockScaledConst(mlir::tosa::ConstOp constOp) {
    if (!constOp) {
        return mlir::failure();
    }

    const auto values = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValues());
    if (!values) {
        return constOp.emitError("block-scaled CONST must contain an inlined dense value attribute");
    }

    const auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(values.getType());
    if (!tensorType || !tensorType.hasStaticShape()) {
        return constOp.emitError("block-scaled CONST must have a statically shaped ranked tensor attribute");
    }
    const auto shape = tensorType.getShape();
    if (shape.empty() || llvm::any_of(shape, [](int64_t dimension) { return dimension <= 0; })) {
        return constOp.emitError("block-scaled CONST must have a non-scalar shape with positive dimensions");
    }
    const auto numElements = mlir::ShapedType::tryGetNumElements(shape);
    if (!numElements || static_cast<uint64_t>(*numElements) > std::numeric_limits<size_t>::max()) {
        return constOp.emitError("block-scaled tensor element count overflows host size");
    }

    const auto blockScaledType = mlir::dyn_cast<mlir::tosa::BlockScaledType>(tensorType.getElementType());
    if (!blockScaledType) {
        return constOp.emitError("CONST is not block-scaled");
    }
    if (!mlir::isa<mlir::Float8E8M0FNUType>(blockScaledType.getScaleType())) {
        return constOp.emitError("block-scaled CONST scale type must be E8M0");
    }
    if (!mlir::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type, mlir::Float6E3M2FNType, mlir::Float6E2M3FNType,
                   mlir::Float4E2M1FNType, mlir::tosa::mxint8Type>(blockScaledType.getValueType())) {
        return constOp.emitError("block-scaled CONST has an unsupported value type");
    }

    const uint32_t blockSize = mlir::tosa::BlockShapeAttr::getBlockShapeValue(blockScaledType.getBlockShape());
    if (blockSize == 0 || !blockScaledType.hasScaleValues()) {
        return constOp.emitError("block-scaled CONST must have a positive block size and scale values");
    }
    if (mlir::failed(mlir::tosa::verifyBlockScaledTensorType(
            tensorType, [&] { return constOp.emitError(); }, /*allowScaleValues=*/true))) {
        return mlir::failure();
    }

    const auto elementCount = static_cast<size_t>(*numElements);
    const mlir::ArrayRef<char> rawValues = values.getRawData();
    if (rawValues.size() != (values.isSplat() ? 1 : elementCount)) {
        return constOp.emitError("block-scaled CONST value payload does not contain one host byte per logical value");
    }

    BlockScaledConstant constant{blockScaledType.getValueType(),
                                 blockScaledType.getScaleType(),
                                 std::vector<int64_t>(shape),
                                 {blockSize},
                                 {},
                                 {}};
    if (values.isSplat()) {
        constant.valueBytes.assign(elementCount, static_cast<uint8_t>(rawValues.front()));
    } else {
        constant.valueBytes.assign(rawValues.begin(), rawValues.end());
    }
    constant.scaleBytes.reserve(blockScaledType.getScaleValues().size());
    for (mlir::Attribute scaleValue : blockScaledType.getScaleValues()) {
        const auto floatValue = mlir::dyn_cast<mlir::FloatAttr>(scaleValue);
        if (!floatValue || floatValue.getType() != constant.scaleType) {
            return constOp.emitError("block-scaled CONST contains a malformed E8M0 scale");
        }
        constant.scaleBytes.push_back(static_cast<uint8_t>(floatValue.getValue().bitcastToAPInt().getZExtValue()));
    }

    if (mlir::failed(packValueBytes(constOp, constant.valueType, constant.valueBytes))) {
        return mlir::failure();
    }
    return constant;
}
#endif

class VGFConstantsPass : public impl::VGFConstantsPassBase<VGFConstantsPass> {
  public:
    VGFConstantsPass() : VGFConstantsPass(std::make_shared<VGFBuilder>()) {}

    explicit VGFConstantsPass(std::shared_ptr<VGFBuilder> VGFBuilder) : _VGFBuilder(std::move(VGFBuilder)) {}

    void runOnOperation() override {
        if (serializeConstants(getOperation()).failed()) {
            llvm::errs() << "Unable to serialize constants\n";
            signalPassFailure();
            return;
        }
    }

  private:
    template <typename T>
    void serializeConstantData(uint32_t graphConstantId, T &attr, ResourceRef resource,
                               int64_t sparsityDimension = -1) {
        mlir::AccessData(attr, [&](const char *data, size_t size) {
            ConstantRef constRef = _VGFBuilder->getEncoder().AddConstant(resource, data, size, sparsityDimension);
            _VGFBuilder->AddConstantBinding(graphConstantId, constRef);
        });
    }

    bool checkIfUnsignedRequired(tosa::ConstOp constOp) {
        auto onlyRescaleOpsWithUnsignedInput = [&constOp](Operation *userOp) {
            if (auto rescaleOp = llvm::dyn_cast_or_null<tosa::RescaleOp>(userOp)) {
                return rescaleOp.getInputUnsigned() && rescaleOp->getOperand(0) == constOp;
            }
            return false;
        };
        return constOp.getType().getElementType().isSignlessInteger() && !constOp->use_empty() &&
               llvm::all_of(constOp->getUsers(), onlyRescaleOpsWithUnsignedInput);
    }

    mlir::LogicalResult serializeConstants(mlir::ModuleOp moduleOp) {

        std::map<uint32_t, tosa::ConstOp> constantsById;
        moduleOp.walk([&constantsById](Operation *op) {
            if (auto constOp = llvm::dyn_cast<tosa::ConstOp>(op)) {
                auto id = constOp->getAttrOfType<IntegerAttr>(tosa::graphARMGraphConstantIdAttrName);
                if (id != nullptr) {
                    constantsById.try_emplace(static_cast<uint32_t>(id.getInt()), constOp);
                }
            }
        });

        for (auto &[id, constOp] : constantsById) {
            const ShapedType type = convertShapedType(constOp.getResult().getType());
            VGFBuilder::VkFormat vkFormat;
            if (VGFBuilder::mlirTypeToVkFormat(type.getElementType(), vkFormat, checkIfUnsignedRequired(constOp))
                    .failed()) {
                return constOp.emitError("unsupported type for tosa.const op: ") << type.getElementType();
            }

            auto format = static_cast<FormatType>(vkFormat);
            ResourceRef resourceRef =
                _VGFBuilder->getEncoder().AddConstantResource(format, flagAnyDynamicDims(type.getShape()), {});

            int64_t sparsityDimension = -1;
            auto attr = constOp->getAttrOfType<IntegerAttr>("constant_2_4_sparse_on_dimension");
            if (attr != nullptr) {
                sparsityDimension = attr.getInt();
            }

            auto attrVal = llvm::dyn_cast<DenseTypedElementsAttr>(constOp.getValuesAttr());
            serializeConstantData(id, attrVal, resourceRef, sparsityDimension);
        }

        return mlir::success();
    }

    std::shared_ptr<VGFBuilder> _VGFBuilder;
};

} // namespace

std::unique_ptr<Pass> createVGFConstantsPass(std::shared_ptr<VGFBuilder> vgfBuilder) {
    return std::make_unique<VGFConstantsPass>(std::move(vgfBuilder));
}

} // namespace mlir::model_converter_passes
