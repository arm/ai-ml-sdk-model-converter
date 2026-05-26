//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: model-converter-opt --tosa-narrow-i64-to-i32="aggressive-rewrite convert-function-boundaries" %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @i64_add
  // CHECK-SAME: tensor<1xi32>
  // CHECK: tosa.add
  // CHECK-SAME: (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  func.func @i64_add(%arg0: tensor<1xi64>, %arg1: tensor<1xi64>) -> tensor<1xi64> {
    %0 = tosa.add %arg0, %arg1 : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
    return %0 : tensor<1xi64>
  }

  // CHECK-LABEL: func.func @i64_const
  // CHECK-SAME: tensor<1xi32>
  // CHECK: "tosa.const"() <{values = dense<1> : tensor<1xi32>}> : () -> tensor<1xi32>
  func.func @i64_const() -> tensor<1xi64> {
    %0 = "tosa.const"() <{values = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
    return %0 : tensor<1xi64>
  }
}
