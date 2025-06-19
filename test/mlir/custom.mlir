//
// SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["input_1"]}, %arg1: tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["input_2"]}) -> (tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["single_custom_op_layer"]}) attributes {tf.entry_function = {inputs = "serving_default_input_1:0,serving_default_input_2:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.custom %arg0, %arg1 {domain_name = "TFL", implementation_attrs = "entry_point\00\04main\00input<0>_binding\00input<0>_type\00\06TENSOR\00input<0>_vkdescriptortype\00%VK_DESCRIPTOR_TYPE_TENSOR_ARM\00input<0>_vkformat\00\14VK_FORMAT_R32_SFLOAT\00input<1>_binding\00input<1>_type\00\06TENSOR\00input<1>_vkdescriptortype\00%VK_DESCRIPTOR_TYPE_TENSOR_ARM\00input<1>_vkformat\00\14VK_FORMAT_R32_SFLOAT\00input_descriptor_set\00is_vkshader\00output<0>_binding\00output<0>_type\00\06TENSOR\00output<0>_vkdescriptortype\00%VK_DESCRIPTOR_TYPE_TENSOR_ARM\00output<0>_vkformat\00\14VK_FORMAT_R32_SFLOAT\00output_descriptor_set\00workgroup_sizes\00\0816,16,16\00\00\11\00\1A\02\0A\02\FB\01\E7\01\A8\01\82\01s\01_\01 \01\FA\00\E7\00\DD\00\CD\00\B8\00x\00Q\00=\00\22\00\02\00\11\005\02\00\00\14\02\F4\01\BD\01\01\00\8C\01l\015\01\00\00\FF\FF\02\00\E5\00\C4\00\8C\00\00\00T\00\14\05\14\14\14\05\14\14\14\05\05\05\14\14\14\05\143%\01", operator_name = "vkshader_pre_custom_op"} : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    return %0 : tensor<1x16x16x16xf32>
  }
}
