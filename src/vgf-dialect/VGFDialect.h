/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#ifndef MLIR_VGF_DIALECT_H
#define MLIR_VGF_DIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

// Interfaces
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "VGFDialect.h.inc"
#include "VGFEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "VGFAttrs.h.inc"

#define GET_OP_CLASSES
#include "VGFOps.h.inc"

#endif // MLIR_VGF_DIALECT_H
