/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::model_converter_passes {

constexpr llvm::StringLiteral vulkanCustomShaderDomainName = "com.arm.VulkanCustomShader";

inline bool isVulkanCustomShaderOp(tosa::CustomOp op) { return op.getDomainName() == vulkanCustomShaderDomainName; }

} // namespace mlir::model_converter_passes
