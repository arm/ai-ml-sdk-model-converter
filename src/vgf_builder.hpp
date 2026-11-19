/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "mlir/Support/LogicalResult.h"

#ifdef BLOCK_SCALED_TENSOR
#    include "mlir/Dialect/Tosa/IR/TosaOps.h"
#endif

#include <vgf/encoder.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace mlsdk::model_converter {

class VGFBuilder {
  public:
    mlsdk::vgflib::Encoder &getEncoder() { return *_encoder; }

    mlir::FailureOr<std::vector<mlsdk::vgflib::GraphConstantBindingRef>>
    getConstantBindings(const std::vector<uint32_t> &graphConstantIds) const {
        std::vector<mlsdk::vgflib::GraphConstantBindingRef> bindings;
        bindings.reserve(graphConstantIds.size());
        for (uint32_t graphConstantId : graphConstantIds) {
            const auto it = _constantRefsByGraphId.find(graphConstantId);
            if (it == _constantRefsByGraphId.end()) {
                return mlir::failure();
            }
            bindings.push_back({graphConstantId, it->second});
        }
        return bindings;
    }

    mlir::LogicalResult addConstantBinding(uint32_t graphConstantId, mlsdk::vgflib::ConstantRef constantRef) {
        return mlir::success(_constantRefsByGraphId.emplace(graphConstantId, constantRef).second);
    }

    // We only support a small handful of Formats for now so redefine the ones
    // we need as it's simpler than adding a dependency on Vulkan-Headers.
    // Note: This won't scale once we support shaders from model to VGF.
    enum VkFormat {
        VK_FORMAT_R8_UINT = 13,
        VK_FORMAT_R8_SINT = 14,
        VK_FORMAT_R16_UINT = 74,
        VK_FORMAT_R16_SINT = 75,
        VK_FORMAT_R16_SFLOAT = 76,
        VK_FORMAT_R32_UINT = 98,
        VK_FORMAT_R32_SINT = 99,
        VK_FORMAT_R32_SFLOAT = 100,
        VK_FORMAT_R64_SINT = 111,
        VK_FORMAT_R8_BOOL_ARM = 1000460000,
        VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM = 1000460001,
        VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM = 1000460002,
        VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM = 1000460003,
        VK_FORMAT_OCPMXFP_32_SFLOAT_BLOCK_FPENCODING_FLOAT6E2M3_ARM = 1000729000,
        VK_FORMAT_OCPMXFP_32_SFLOAT_BLOCK_FPENCODING_FLOAT6E3M2_ARM = 1000729001,
        VK_FORMAT_OCPMXFP_32_SFLOAT_BLOCK_FPENCODING_FLOAT4E2M1_ARM = 1000729002,
        VK_FORMAT_R8_SFLOAT_FPENCODING_MXINT8_ARM = 1000729004,
    };

    static mlir::LogicalResult mlirTypeToVkFormat(mlir::Type mlirType, VkFormat &format, bool useUnsignedForSignless) {
        if (mlirType.isInteger(1)) {
            format = VkFormat::VK_FORMAT_R8_BOOL_ARM;
        } else if (mlirType.isSignedInteger() || (mlirType.isSignlessInteger() && !useUnsignedForSignless)) {
            switch (mlirType.getIntOrFloatBitWidth()) {
            case 4:
            case 8:
                format = VkFormat::VK_FORMAT_R8_SINT;
                break;
            case 16:
                format = VkFormat::VK_FORMAT_R16_SINT;
                break;
            case 32:
                format = VkFormat::VK_FORMAT_R32_SINT;
                break;
            case 48:
            case 64:
                format = VkFormat::VK_FORMAT_R64_SINT;
                break;
            default:
                return mlir::failure();
            }
        } else if (mlirType.isUnsignedInteger() || (mlirType.isSignlessInteger() && useUnsignedForSignless)) {
            switch (mlirType.getIntOrFloatBitWidth()) {
            case 8:
                format = VkFormat::VK_FORMAT_R8_UINT;
                break;
            case 16:
                format = VkFormat::VK_FORMAT_R16_UINT;
                break;
            case 32:
                format = VkFormat::VK_FORMAT_R32_UINT;
                break;
            default:
                return mlir::failure();
            }
        } else if (mlirType.isF16()) {
            format = VkFormat::VK_FORMAT_R16_SFLOAT;
        } else if (mlirType.isF32()) {
            format = VkFormat::VK_FORMAT_R32_SFLOAT;
        } else if (mlirType.isBF16()) {
            format = VkFormat::VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM;
        } else if (mlirType.isF8E4M3FN()) {
            format = VkFormat::VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM;
        } else if (mlirType.isF8E5M2()) {
            format = VkFormat::VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM;
        } else if (llvm::isa<mlir::Float6E2M3FNType>(mlirType)) {
            format = VkFormat::VK_FORMAT_OCPMXFP_32_SFLOAT_BLOCK_FPENCODING_FLOAT6E2M3_ARM;
        } else if (llvm::isa<mlir::Float6E3M2FNType>(mlirType)) {
            format = VkFormat::VK_FORMAT_OCPMXFP_32_SFLOAT_BLOCK_FPENCODING_FLOAT6E3M2_ARM;
        } else if (llvm::isa<mlir::Float4E2M1FNType>(mlirType)) {
            format = VkFormat::VK_FORMAT_OCPMXFP_32_SFLOAT_BLOCK_FPENCODING_FLOAT4E2M1_ARM;
#ifdef BLOCK_SCALED_TENSOR
        } else if (llvm::isa<mlir::OCPInt8Type, mlir::tosa::mxint8Type>(mlirType)) {
            format = VkFormat::VK_FORMAT_R8_SFLOAT_FPENCODING_MXINT8_ARM;
#endif
        } else {
            return mlir::failure();
        }
        return mlir::success();
    }

  private:
    std::unique_ptr<mlsdk::vgflib::Encoder> _encoder = mlsdk::vgflib::CreateEncoder(0);
    std::map<uint32_t, mlsdk::vgflib::ConstantRef> _constantRefsByGraphId;
};

} // namespace mlsdk::model_converter
