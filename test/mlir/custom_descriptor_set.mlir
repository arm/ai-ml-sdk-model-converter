//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

module attributes {tf_saved_model.semantics, tosa.description = "TOSA FBS Converted", tosa.fbs_version = "1.1.0d"} {
  func.func @main(%arg0: tensor<10xf32> {tf_saved_model.index_path = ["input_0"]}, %arg1: tensor<10xf32> {tf_saved_model.index_path = ["input_1"]}) -> (tensor<10xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "tosa_deserialized_input_0:0,tosa_deserialized_input_1:0", outputs = "tosa_deserialized_output_0:0"}, tf_saved_model.exported_names = ["tosa_deserialized"]} {
    %0 = tosa.add %arg0, %arg1 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22: \22main\22, \22workgroup_sizes\22: [64, 1, 1], \22is_vkshader\22: true, \22push_constants\22: \22\22, \22input_0_binding\22: 0, \22output_0_binding\22: 1, \22input_0_type\22: \22Tensor\22, \22output_0_type\22: \22Tensor\22, \22input_0_vkdescriptortype\22: \22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22, \22output_0_vkdescriptortype\22: \22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22, \22input_0_descriptorset\22: 7, \22output_0_descriptorset\22: 9, \22input_0_vkformat\22: \22VK_FORMAT_R32_SFLOAT\22, \22output_0_vkformat\22: \22VK_FORMAT_R32_SFLOAT\22}", operator_name = "thribrary.threee_pleee"} : (tensor<10xf32>) -> tensor<10xf32>
    return %1 : tensor<10xf32>
  }
}
