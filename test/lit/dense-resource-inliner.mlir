//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: model-converter-opt --dense-resource-inliner %s | FileCheck %s

// CHECK-LABEL: func.func @plain_constant()
// CHECK: tosa.const values(dense<[-7, 42]> : tensor<2xi32>) : () -> tensor<2xi32>
func.func @plain_constant() -> tensor<2xi32> {
  %0 = tosa.const values(dense_resource<plain_values> : tensor<2xi32>) : () -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

{-#
  dialect_resources: {
    builtin: {
      plain_values: "0x04000000F9FFFFFF2A000000"
    }
  }
#-}
