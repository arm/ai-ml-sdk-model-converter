/*
* SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/
#ifndef MLIR_VGF_DIALECT_ENUMS_TD
#define MLIR_VGF_DIALECT_ENUMS_TD

include "src/vgf-dialect/VGFDialect.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// VGF Dialect Enums
//===----------------------------------------------------------------------===//

def SegmentTypeEnumAttr : I32EnumAttr<"SegmentTypeEnum", "Type of segment",
  [
    I32EnumAttrCase<"GRAPH", 0, "graph">,
    I32EnumAttrCase<"COMPUTE", 1, "compute">
  ]> {
    let cppNamespace = "::mlir::vgf";
  }

#endif // MLIR_VGF_DIALECT_ENUMS_TD
