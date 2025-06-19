/*
* SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/
#ifndef MLIR_VGF_DIALECT_TD
#define MLIR_VGF_DIALECT_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// VGF Dialect.
//===----------------------------------------------------------------------===//

def VGF_Dialect : Dialect {
  let name = "vgf";
  let cppNamespace = "::mlir::vgf";

  let summary = [{
    TODO
  }];

  let description = [{
    TODO
  }];

}

#endif  // MLIR_VGF_DIALECT_TD
