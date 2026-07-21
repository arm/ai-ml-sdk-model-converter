/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "model_partition_common.hpp"

#include "include/custom_op_domains.hpp"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"

namespace mlir::model_converter_passes {

bool isCompileTimeTosaConstant(Operation *op) { return llvm::isa_and_nonnull<tosa::ConstOp, tosa::ConstShapeOp>(op); }

bool isVulkanCustomShaderOperation(Operation *op) {
    auto customOp = llvm::dyn_cast_or_null<tosa::CustomOp>(op);
    return customOp && isVulkanCustomShaderOp(customOp);
}

} // namespace mlir::model_converter_passes
