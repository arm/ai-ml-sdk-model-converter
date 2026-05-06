//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

module {
  func.func @main(%arg0: tensor<1x8x8x4xf32> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x8x8x4xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "tosa_deserialized_input_0:0", outputs = "tosa_deserialized_output_0:0"}, tf_saved_model.exported_names = ["tosa_deserialized"]} {
    %0 = tosa.custom %arg0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\"entry_point\":\"main\",\"input_0_binding\":0,\"input_0_descriptorset\":0,\"input_0_type\":\"Image\",\"input_0_vkdescriptortype\":\"VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER\",\"input_0_vkformat\":\"VK_FORMAT_R32G32B32A32_SFLOAT\",\"input_0_sampler\":{\"min_filter\":\"VK_FILTER_LINEAR\",\"mag_filter\":\"VK_FILTER_LINEAR\",\"address_mode_u\":\"VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER\",\"address_mode_v\":\"VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER\",\"border_color\":\"VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK\"},\"is_vkshader\":true,\"output_0_binding\":1,\"output_0_descriptorset\":0,\"output_0_type\":\"Image\",\"output_0_vkdescriptortype\":\"VK_DESCRIPTOR_TYPE_STORAGE_IMAGE\",\"output_0_vkformat\":\"VK_FORMAT_R32G32B32A32_SFLOAT\",\"workgroup_sizes\":[8,8,1]}", operator_name = "test_image_sampler_shader"} : (tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32>
    return %0 : tensor<1x8x8x4xf32>
  }
}
