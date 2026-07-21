/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#pragma once

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::model_converter_passes {

bool isCompileTimeTosaConstant(Operation *op);
bool isVulkanCustomShaderOperation(Operation *op);

} // namespace mlir::model_converter_passes
