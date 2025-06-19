/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include <vgf/encoder.hpp>

namespace mlsdk::model_converter {

class VGFBuilder {
  public:
    ~VGFBuilder(){};
    VGFBuilder() : _encoder(mlsdk::vgflib::CreateEncoder(0)) {}
    std::shared_ptr<mlsdk::vgflib::Encoder> getEncoder() const { return _encoder; }
    const std::vector<mlsdk::vgflib::ConstantRef> &getConstantRefs() const { return _constantRefs; }

    void AddConstantRef(mlsdk::vgflib::ConstantRef constRef) { _constantRefs.emplace_back(constRef); }

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
        } else {
            return mlir::failure();
        }
        return mlir::success();
    }

  private:
    std::shared_ptr<mlsdk::vgflib::Encoder> _encoder = {};
    std::vector<mlsdk::vgflib::ConstantRef> _constantRefs = {};
};

} // namespace mlsdk::model_converter
