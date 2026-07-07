/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "model_partition_common.hpp"

#include "include/custom_op_domains.hpp"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace mlir::model_converter_passes {
namespace {

std::optional<unsigned> getIntegerElementBitWidth(Value value) {
    auto type = llvm::dyn_cast<ShapedType>(value.getType());
    if (!type || !type.getElementType().isInteger()) {
        return std::nullopt;
    }
    return type.getElementType().getIntOrFloatBitWidth();
}

std::optional<DeferredMaterializationInfo> getRescaleDeferredMaterializationInfo(tosa::RescaleOp rescaleOp) {
    const auto inputWidth = getIntegerElementBitWidth(rescaleOp.getInput());
    const auto outputWidth = getIntegerElementBitWidth(rescaleOp->getResult(0));
    if (!inputWidth || !outputWidth || *inputWidth >= *outputWidth) {
        return std::nullopt;
    }

    DeferredMaterializationInfo info;
    Value input = rescaleOp.getInput();
    if (isCompileTimeTosaConstant(input.getDefiningOp())) {
        info.constantsToClone.push_back(input);
    } else {
        info.runtimeInputs.push_back(input);
    }

    for (Value operand : rescaleOp->getOperands().drop_front()) {
        if (!isCompileTimeTosaConstant(operand.getDefiningOp())) {
            return std::nullopt;
        }
        info.constantsToClone.push_back(operand);
    }

    return info;
}

} // namespace

bool isCompileTimeTosaConstant(Operation *op) { return llvm::isa_and_nonnull<tosa::ConstOp, tosa::ConstShapeOp>(op); }

bool isVulkanCustomShaderOperation(Operation *op) {
    auto customOp = llvm::dyn_cast_or_null<tosa::CustomOp>(op);
    return customOp && isVulkanCustomShaderOp(customOp);
}

std::optional<DeferredMaterializationInfo> getDeferredMaterializationInfo(Operation *op) {
    if (auto rescaleOp = llvm::dyn_cast_or_null<tosa::RescaleOp>(op)) {
        return getRescaleDeferredMaterializationInfo(rescaleOp);
    }

    return std::nullopt;
}

} // namespace mlir::model_converter_passes
