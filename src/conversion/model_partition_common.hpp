/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::model_converter_passes {

struct DeferredMaterializationInfo {
    SmallVector<Value> runtimeInputs;
    SmallVector<Value> constantsToClone;
};

bool isCompileTimeTosaConstant(Operation *op);
bool isVulkanCustomShaderOperation(Operation *op);
std::optional<DeferredMaterializationInfo> getDeferredMaterializationInfo(Operation *op);

} // namespace mlir::model_converter_passes
