//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: model-converter-opt --model-partition-marking --model-partitioning %s | FileCheck %s

// CHECK-LABEL: vgf.sequence @main
// CHECK: func.func @graph_partition_0() -> (tensor<1x32x32xf32>, tensor<1x32x32xf32>)
// CHECK: %[[REAL:.*]], %[[IMAG:.*]] = tosa.fft2d
// CHECK: return %[[REAL]], %[[IMAG]] : tensor<1x32x32xf32>, tensor<1x32x32xf32>
// CHECK: %[[SEGMENT_EXEC:.*]]:2 = vgf.run_segment @graph_partition_0 () -> (tensor<1x32x32xf32>, tensor<1x32x32xf32>)
// CHECK: vgf.sequence_output %[[SEGMENT_EXEC]]#0, %[[SEGMENT_EXEC]]#1 : tensor<1x32x32xf32>, tensor<1x32x32xf32>
func.func @main() -> (tensor<1x32x32xf32> {tf_saved_model.index_path = ["output_0"]}, tensor<1x32x32xf32> {tf_saved_model.index_path = ["output_1"]}) attributes {tf.entry_function = {outputs = "output_0,output_1"}} {
  %real = "tosa.const"() {values = dense<0.0> : tensor<1x32x32xf32>} : () -> tensor<1x32x32xf32>
  %imag = "tosa.const"() {values = dense<0.0> : tensor<1x32x32xf32>} : () -> tensor<1x32x32xf32>
  %output_real, %output_imag = tosa.fft2d %real, %imag {inverse = true} : (tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> (tensor<1x32x32xf32>, tensor<1x32x32xf32>)
  return %output_real, %output_imag : tensor<1x32x32xf32>, tensor<1x32x32xf32>
}
