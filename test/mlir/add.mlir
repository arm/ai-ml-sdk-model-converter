//
// SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

module {
  func.func @main(%arg0: tensor<f32> {tf_saved_model.index_path = ["input_1"]}, %arg1: tensor<f32> {tf_saved_model.index_path = ["input_2"]}) -> (tensor<f32> {tf_saved_model.index_path = ["single_custom_op_layer"]}) attributes {tf.entry_function = {inputs = "serving_default_input_1:0,serving_default_input_2:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.add %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}
