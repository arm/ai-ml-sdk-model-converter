//
// SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

module {
  func.func @main(%arg0: tensor<1x16x16x16xi8> {tf_saved_model.index_path = ["input_1"]}, %arg1: tensor<1x8x8x16xi8> {tf_saved_model.index_path = ["input_3"]}, %arg2: tensor<1x16x16x16xi8> {tf_saved_model.index_path = ["input_2"]}) -> (tensor<1x8x8x16xi8> {tf_saved_model.index_path = ["tf.quantization.fake_quant_with_min_max_vars_4"]}) attributes {tf.entry_function = {inputs = "input_0,input_1,input_2", outputs = "output_0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.custom %arg0, %arg2 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\"entry_point\":\"main\",\"input_0_binding\":0,\"input_0_descriptorset\":0,\"input_0_type\":\"TENSOR\",\"input_0_vkdescriptortype\":\"VK_DESCRIPTOR_TYPE_TENSOR_ARM\",\"input_0_vkformat\":\"VK_FORMAT_R8_SINT\",\"input_1_binding\":1,\"input_1_descriptorset\":0,\"input_1_type\":\"TENSOR\",\"input_1_vkdescriptortype\":\"VK_DESCRIPTOR_TYPE_TENSOR_ARM\",\"input_1_vkformat\":\"VK_FORMAT_R8_SINT\",\"is_vkshader\":true,\"output_0_binding\":2,\"output_0_descriptorset\":0,\"output_0_type\":\"TENSOR\",\"output_0_vkdescriptortype\":\"VK_DESCRIPTOR_TYPE_TENSOR_ARM\",\"output_0_vkformat\":\"VK_FORMAT_R8_SINT\",\"workgroup_sizes\":[16,16,16]}", operator_name = "test_placeholder_shader_0"} : (tensor<1x16x16x16xi8>, tensor<1x16x16x16xi8>) -> tensor<1x16x16x16xi8>
    %1 = tosa.max_pool2d %0 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x16x16x16xi8>) -> tensor<1x8x8x16xi8>
    %2 = tosa.custom %arg1, %1 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\"entry_point\":\"main\",\"input_0_binding\":0,\"input_0_descriptorset\":0,\"input_0_type\":\"TENSOR\",\"input_0_vkdescriptortype\":\"VK_DESCRIPTOR_TYPE_TENSOR_ARM\",\"input_0_vkformat\":\"VK_FORMAT_R8_SINT\",\"input_1_binding\":1,\"input_1_descriptorset\":0,\"input_1_type\":\"TENSOR\",\"input_1_vkdescriptortype\":\"VK_DESCRIPTOR_TYPE_TENSOR_ARM\",\"input_1_vkformat\":\"VK_FORMAT_R8_SINT\",\"is_vkshader\":true,\"output_0_binding\":2,\"output_0_descriptorset\":0,\"output_0_type\":\"TENSOR\",\"output_0_vkdescriptortype\":\"VK_DESCRIPTOR_TYPE_TENSOR_ARM\",\"output_0_vkformat\":\"VK_FORMAT_R8_SINT\",\"workgroup_sizes\":[8,8,16]}", operator_name = "test_placeholder_shader_1"} : (tensor<1x8x8x16xi8>, tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8>
    return %2 : tensor<1x8x8x16xi8>
  }
}
