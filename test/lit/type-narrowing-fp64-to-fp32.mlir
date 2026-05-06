//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: model-converter --input %s --output %t --tosa-flatbuffer --dump-mlir 2>&1 | FileCheck %s

module {
  // CHECK-LABEL: IR Dump After TosaNarrowF64ToF32Pass
  // CHECK-SAME: @f64_add
  // CHECK-LABEL: func.func @f64_add
  // CHECK-SAME: tensor<1xf32>
  // CHECK: tosa.add
  // CHECK-SAME: (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  func.func @f64_add(%arg0: tensor<1xf64>, %arg1: tensor<1xf64>) -> tensor<1xf64> {
    %0 = tosa.add %arg0, %arg1 : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
    return %0 : tensor<1xf64>
  }

  // CHECK-LABEL: IR Dump After TosaNarrowF64ToF32Pass
  // CHECK-SAME: @f64_const
  // CHECK-LABEL: func.func @f64_const
  // CHECK-SAME: tensor<1xf32>
  // CHECK: "tosa.const"() <{values = dense<1.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
  func.func @f64_const() -> tensor<1xf64> {
    %0 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1xf64>}> : () -> tensor<1xf64>
    return %0 : tensor<1xf64>
  }
}
