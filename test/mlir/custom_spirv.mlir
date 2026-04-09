//
// SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

module {
  func.func @main(%arg0: tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["input_1"]}, %arg1: tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["input_2"]}) -> (tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["single_custom_op_layer"]}) attributes {tf.entry_function = {inputs = "serving_default_input_1:0,serving_default_input_2:0", outputs = "PartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.custom %arg0, %arg1 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\"entry_point\":\"main\",\"input_0_binding\":0,\"input_0_descriptorset\":0,\"input_0_type\":\"TENSOR\",\"input_0_vkdescriptortype\":\"VK_DESCRIPTOR_TYPE_TENSOR_ARM\",\"input_0_vkformat\":\"VK_FORMAT_R32_SFLOAT\",\"input_1_binding\":1,\"input_1_descriptorset\":0,\"input_1_type\":\"TENSOR\",\"input_1_vkdescriptortype\":\"VK_DESCRIPTOR_TYPE_TENSOR_ARM\",\"input_1_vkformat\":\"VK_FORMAT_R32_SFLOAT\",\"is_vkshader\":true,\"output_0_binding\":2,\"output_0_descriptorset\":0,\"output_0_type\":\"TENSOR\",\"output_0_vkdescriptortype\":\"VK_DESCRIPTOR_TYPE_TENSOR_ARM\",\"output_0_vkformat\":\"VK_FORMAT_R32_SFLOAT\",\"shader_language\":\"SPIR-V\",\"shader_code\":\"AwIjBwAAAQAAAAAAAAAAAA==\",\"workgroup_sizes\":[16,16,16]}", operator_name = "test_placeholder_shader_spirv"} : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
    return %0 : tensor<1x16x16x16xf32>
  }
}
