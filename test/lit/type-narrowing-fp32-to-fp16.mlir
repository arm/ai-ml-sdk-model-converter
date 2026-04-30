//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: model-converter-opt --type-narrowing="mode=full" %s | FileCheck %s --check-prefix=FULL
// RUN: model-converter-opt --type-narrowing="mode=full_preserve_io" %s | FileCheck %s --check-prefix=PRESERVE-IO
// RUN: model-converter-opt --type-narrowing="mode=partial" %s | FileCheck %s --check-prefix=PARTIAL

module {
  // FULL: func.func @main(%{{.*}}: tensor<1xf16>, %{{.*}}: tensor<1xf16>) -> tensor<1xf16>
  // FULL: "tosa.const"() <{values = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  // FULL: tosa.add
  // FULL-SAME: (tensor<1xf16>, tensor<1xf16>) -> tensor<1xf16>
  // FULL: tosa.abs
  // FULL-SAME: (tensor<1xf16>) -> tensor<1xf16>
  // FULL: return
  // FULL-SAME: tensor<1xf16>
  //
  // PRESERVE-IO: func.func @main(%[[ARG0:.*]]: tensor<1xf32>, %[[ARG1:.*]]: tensor<1xf32>) -> tensor<1xf32>
  // PRESERVE-IO-DAG: %[[ARG1_CAST:.*]] = tosa.cast %[[ARG1]] : (tensor<1xf32>) -> tensor<1xf16>
  // PRESERVE-IO-DAG: %[[ARG0_CAST:.*]] = tosa.cast %[[ARG0]] : (tensor<1xf32>) -> tensor<1xf16>
  // PRESERVE-IO: %[[CONST:.*]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
  // PRESERVE-IO: %[[ADD0:.*]] = tosa.add %[[ARG0_CAST]], %[[CONST]]
  // PRESERVE-IO-SAME: (tensor<1xf16>, tensor<1xf16>) -> tensor<1xf16>
  // PRESERVE-IO: %[[ABS:.*]] = tosa.abs %[[ADD0]] : (tensor<1xf16>) -> tensor<1xf16>
  // PRESERVE-IO: %[[ADD1:.*]] = tosa.add %[[ABS]], %[[ARG1_CAST]]
  // PRESERVE-IO-SAME: (tensor<1xf16>, tensor<1xf16>) -> tensor<1xf16>
  // PRESERVE-IO: %[[RET_CAST:.*]] = tosa.cast %[[ADD1]] : (tensor<1xf16>) -> tensor<1xf32>
  // PRESERVE-IO: return %[[RET_CAST]] : tensor<1xf32>
  func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
    %0 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.add %arg0, %0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %2 = tosa.abs %1 : (tensor<1xf32>) -> tensor<1xf32>
    %3 = tosa.add %2, %arg1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    return %3 : tensor<1xf32>
  }

  // FULL-LABEL: func.func @avg_pool
  // FULL: tosa.avg_pool2d
  // FULL-SAME: acc_type = f16
  // FULL-SAME: (tensor<1x4x4x1xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x3x3x1xf16>
  //
  // PRESERVE-IO-LABEL: func.func @avg_pool
  // PRESERVE-IO-SAME: tensor<1x4x4x1xf32>
  // PRESERVE-IO-SAME: tensor<1x3x3x1xf32>
  // PRESERVE-IO: tosa.cast
  // PRESERVE-IO-SAME: (tensor<1x4x4x1xf32>) -> tensor<1x4x4x1xf16>
  // PRESERVE-IO: tosa.avg_pool2d
  // PRESERVE-IO-SAME: acc_type = f16
  // PRESERVE-IO-SAME: (tensor<1x4x4x1xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x3x3x1xf16>
  // PRESERVE-IO: tosa.cast
  // PRESERVE-IO-SAME: (tensor<1x3x3x1xf16>) -> tensor<1x3x3x1xf32>
  //
  // PARTIAL-LABEL: func.func @avg_pool
  // PARTIAL: tosa.avg_pool2d
  // PARTIAL-SAME: acc_type = f32
  // PARTIAL-SAME: (tensor<1x4x4x1xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x3x3x1xf16>
  func.func @avg_pool(%arg0: tensor<1x4x4x1xf32>) -> tensor<1x3x3x1xf32> {
    %0 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = tosa.avg_pool2d %arg0, %0, %1 {acc_type = f32, kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x3x3x1xf32>
    return %2 : tensor<1x3x3x1xf32>
  }

  // FULL-LABEL: func.func @conv
  // FULL: tosa.conv2d
  // FULL-SAME: acc_type = f16
  // FULL-SAME: (tensor<1x4x4x1xf16>, tensor<1x3x3x1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x2x2x1xf16>
  //
  // PRESERVE-IO-LABEL: func.func @conv
  // PRESERVE-IO-SAME: tensor<1x4x4x1xf32>
  // PRESERVE-IO-SAME: tensor<1x2x2x1xf32>
  // PRESERVE-IO: tosa.conv2d
  // PRESERVE-IO-SAME: acc_type = f16
  // PRESERVE-IO-SAME: (tensor<1x4x4x1xf16>, tensor<1x3x3x1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x2x2x1xf16>
  // PRESERVE-IO: tosa.cast
  // PRESERVE-IO-SAME: (tensor<1x2x2x1xf16>) -> tensor<1x2x2x1xf32>
  //
  // PARTIAL-LABEL: func.func @conv
  // PARTIAL: tosa.conv2d
  // PARTIAL-SAME: acc_type = f32
  // PARTIAL-SAME: (tensor<1x4x4x1xf16>, tensor<1x3x3x1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x2x2x1xf16>
  func.func @conv(%arg0: tensor<1x4x4x1xf32>) -> tensor<1x2x2x1xf32> {
    %0 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x3x3x1xf32>}> : () -> tensor<1x3x3x1xf32>
    %1 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %4 = tosa.conv2d %arg0, %0, %1, %2, %3 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x1xf32>, tensor<1x3x3x1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x2x2x1xf32>
    return %4 : tensor<1x2x2x1xf32>
  }

  // FULL-LABEL: func.func @depthwise_conv
  // FULL: tosa.depthwise_conv2d
  // FULL-SAME: acc_type = f16
  // FULL-SAME: (tensor<1x4x4x1xf16>, tensor<3x3x1x1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x2x2x1xf16>
  //
  // PRESERVE-IO-LABEL: func.func @depthwise_conv
  // PRESERVE-IO-SAME: tensor<1x4x4x1xf32>
  // PRESERVE-IO-SAME: tensor<1xf32>
  // PRESERVE-IO-SAME: tensor<1x2x2x1xf32>
  // PRESERVE-IO: tosa.depthwise_conv2d
  // PRESERVE-IO-SAME: acc_type = f16
  // PRESERVE-IO-SAME: (tensor<1x4x4x1xf16>, tensor<3x3x1x1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x2x2x1xf16>
  // PRESERVE-IO: tosa.cast
  // PRESERVE-IO-SAME: (tensor<1x2x2x1xf16>) -> tensor<1x2x2x1xf32>
  //
  // PARTIAL-LABEL: func.func @depthwise_conv
  // PARTIAL: tosa.depthwise_conv2d
  // PARTIAL-SAME: acc_type = f32
  // PARTIAL-SAME: (tensor<1x4x4x1xf16>, tensor<3x3x1x1xf16>, tensor<1xf16>, tensor<1xf16>, tensor<1xf16>) -> tensor<1x2x2x1xf16>
  func.func @depthwise_conv(%arg0: tensor<1x4x4x1xf32>, %arg1: tensor<1xf32>) -> tensor<1x2x2x1xf32> {
    %0 = "tosa.const"() <{values = dense<1.000000e+00> : tensor<3x3x1x1xf32>}> : () -> tensor<3x3x1x1xf32>
    %1 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = tosa.depthwise_conv2d %arg0, %0, %arg1, %1, %2 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x4x4x1xf32>, tensor<3x3x1x1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x2x2x1xf32>
    return %3 : tensor<1x2x2x1xf32>
  }
}
