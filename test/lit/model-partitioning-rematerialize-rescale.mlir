//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: model-converter-opt --split-input-file --model-partition-marking --model-partitioning %s | FileCheck %s

func.func @rescale_rematerialized_after_shader(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @rescale_rematerialized_after_shader
  // CHECK: func.func @graph_partition_0(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: return
  // CHECK: vgf.segment @graph_partition_1
  // CHECK: func.func @graph_partition_2(%{{[^:]+}}: tensor<4xi8>, %{{[^:]+}}: tensor<4xi8>) -> tensor<4xi32>
  // CHECK: tosa.rescale
  // CHECK: tosa.rescale
  // CHECK: tosa.add
  %multiplier = "tosa.const"() {values = dense<[1]> : tensor<1xi32>} : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<[0]> : tensor<1xi8>} : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<[0]> : tensor<1xi8>} : () -> tensor<1xi8>
  %flow_output_zp = "tosa.const"() {values = dense<[0]> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<[0]> : tensor<1xi32>} : () -> tensor<1xi32>
  %flow = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %flow_output_zp {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<4xi8>
  %old_flow_i32 = tosa.rescale %flow, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<4xi32>
  %shader = tosa.custom %flow {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "rescale_remat_shader"} : (tensor<4xi8>) -> tensor<4xi8>
  %shader_i32 = tosa.rescale %shader, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<4xi32>
  %out = tosa.add %shader_i32, %old_flow_i32 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %out : tensor<4xi32>
}

// -----

func.func @rescale_compute_use_preserved_graph_use_rematerialized(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @rescale_compute_use_preserved_graph_use_rematerialized
  // CHECK: func.func @graph_partition_0(%{{[^:]+}}: tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi32>)
  // CHECK: tosa.rescale
  // CHECK: vgf.segment @graph_partition_1
  // CHECK: vgf.run_segment @graph_partition_1 : (%{{[^)]*}}) (tensor<4xi32>) -> tensor<4xi32>
  // CHECK: func.func @graph_partition_2(%{{[^:]+}}: tensor<4xi32>, %{{[^:]+}}: tensor<4xi8>) -> tensor<4xi32>
  // CHECK: tosa.rescale
  // CHECK: tosa.add
  %multiplier = "tosa.const"() {values = dense<[1]> : tensor<1xi32>} : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<[0]> : tensor<1xi8>} : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<[0]> : tensor<1xi8>} : () -> tensor<1xi8>
  %flow_output_zp = "tosa.const"() {values = dense<[0]> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<[0]> : tensor<1xi32>} : () -> tensor<1xi32>
  %flow = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %flow_output_zp {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<4xi8>
  %old_flow_i32 = tosa.rescale %flow, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<4xi32>
  %shader = tosa.custom %old_flow_i32 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R32_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R32_SINT\22}", operator_name = "rescale_i32_shader"} : (tensor<4xi32>) -> tensor<4xi32>
  %out = tosa.add %shader, %old_flow_i32 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %out : tensor<4xi32>
}

// -----

func.func @rescale_model_output_preserved(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi32> {tf_saved_model.index_path = ["output_0"]}, tensor<4xi32> {tf_saved_model.index_path = ["output_1"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0,output_1"}} {
  // CHECK-LABEL: vgf.sequence @rescale_model_output_preserved
  // CHECK: func.func @graph_partition_0(%{{[^:]+}}: tensor<4xi8>) -> (tensor<4xi32>, tensor<4xi8>)
  // CHECK: func.func @graph_partition_2(%{{[^:]+}}: tensor<4xi8>, %{{[^:]+}}: tensor<4xi8>) -> tensor<4xi32>
  // CHECK: vgf.sequence_output %{{.*}}, %{{.*}} : tensor<4xi32>, tensor<4xi32>
  %multiplier = "tosa.const"() {values = dense<[1]> : tensor<1xi32>} : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<[0]> : tensor<1xi8>} : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<[0]> : tensor<1xi8>} : () -> tensor<1xi8>
  %flow_output_zp = "tosa.const"() {values = dense<[0]> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<[0]> : tensor<1xi32>} : () -> tensor<1xi32>
  %flow = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %flow_output_zp {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<4xi8>
  %old_flow_i32 = tosa.rescale %flow, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<4xi32>
  %shader = tosa.custom %flow {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "rescale_output_shader"} : (tensor<4xi8>) -> tensor<4xi8>
  %shader_i32 = tosa.rescale %shader, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<4xi32>
  %out = tosa.add %shader_i32, %old_flow_i32 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %out, %old_flow_i32 : tensor<4xi32>, tensor<4xi32>
}

// -----

func.func @non_constant_rescale_params_not_rematerialized(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}, %arg1: tensor<1xi32> {tf_saved_model.index_path = ["input_1"]}) -> (tensor<4xi32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0,input_1", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @non_constant_rescale_params_not_rematerialized
  // CHECK: func.func @graph_partition_0(%{{[^:]+}}: tensor<4xi8>, %{{[^:]+}}: tensor<1xi32>) -> (tensor<4xi8>, tensor<4xi32>)
  // CHECK: func.func @graph_partition_2(%{{[^:]+}}: tensor<4xi8>, %{{[^:]+}}: tensor<1xi32>, %{{[^:]+}}: tensor<4xi32>) -> tensor<4xi32>
  %shift = "tosa.const"() {values = dense<[0]> : tensor<1xi8>} : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<[0]> : tensor<1xi8>} : () -> tensor<1xi8>
  %flow_output_zp = "tosa.const"() {values = dense<[0]> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<[0]> : tensor<1xi32>} : () -> tensor<1xi32>
  %flow = tosa.rescale %arg0, %arg1, %shift, %input_zp, %flow_output_zp {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<4xi8>
  %old_flow_i32 = tosa.rescale %flow, %arg1, %shift, %input_zp, %output_zp {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<4xi32>
  %shader = tosa.custom %flow {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "non_constant_rescale_params_shader"} : (tensor<4xi8>) -> tensor<4xi8>
  %shader_i32 = tosa.rescale %shader, %arg1, %shift, %input_zp, %output_zp {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<4xi32>
  %out = tosa.add %shader_i32, %old_flow_i32 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %out : tensor<4xi32>
}

// -----

func.func @rescale_rematerialized_in_each_consuming_graph_partition(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @rescale_rematerialized_in_each_consuming_graph_partition
  // CHECK: func.func @graph_partition_0(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: tosa.reverse
  // CHECK-NEXT: return
  // CHECK: vgf.segment @graph_partition_1
  // CHECK: func.func @graph_partition_2
  // CHECK: tosa.rescale
  // CHECK: tosa.rescale
  // CHECK: tosa.add
  // CHECK: vgf.segment @graph_partition_3
  // CHECK: func.func @graph_partition_4
  // CHECK: tosa.rescale
  // CHECK: tosa.add
  %multiplier = "tosa.const"() {values = dense<[1]> : tensor<1xi32>} : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<[0]> : tensor<1xi8>} : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<[0]> : tensor<1xi8>} : () -> tensor<1xi8>
  %output_zp = "tosa.const"() {values = dense<[0]> : tensor<1xi32>} : () -> tensor<1xi32>
  %flow = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<4xi8>) -> tensor<4xi8>
  %flow_i32 = tosa.rescale %flow, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<4xi32>
  %shader_0 = tosa.custom %flow {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "first_shader"} : (tensor<4xi8>) -> tensor<4xi8>
  %shader_0_i32 = tosa.rescale %shader_0, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<4xi32>
  %graph_2 = tosa.add %shader_0_i32, %flow_i32 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  %shader_1 = tosa.custom %graph_2 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R32_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R32_SINT\22}", operator_name = "second_shader"} : (tensor<4xi32>) -> tensor<4xi32>
  %out = tosa.add %shader_1, %flow_i32 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %out : tensor<4xi32>
}

// -----

func.func @equal_width_rescale_not_rematerialized(%arg0: tensor<4xi16> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi16> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @equal_width_rescale_not_rematerialized
  // CHECK: func.func @graph_partition_0(%{{[^:]+}}: tensor<4xi16>) -> tensor<4xi16>
  // CHECK: tosa.rescale
  // CHECK: vgf.segment @graph_partition_1
  // CHECK: func.func @graph_partition_2(%{{[^:]+}}: tensor<4xi16>, %{{[^:]+}}: tensor<4xi16>) -> tensor<4xi16>
  // CHECK-NOT: tosa.rescale
  // CHECK: tosa.add
  %multiplier = "tosa.const"() {values = dense<[1]> : tensor<1xi32>} : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<[0]> : tensor<1xi8>} : () -> tensor<1xi8>
  %zero_point = "tosa.const"() {values = dense<[0]> : tensor<1xi16>} : () -> tensor<1xi16>
  %rescaled = tosa.rescale %arg0, %multiplier, %shift, %zero_point, %zero_point {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<4xi16>, tensor<1xi32>, tensor<1xi8>, tensor<1xi16>, tensor<1xi16>) -> tensor<4xi16>
  %shader = tosa.custom %arg0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R16_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R16_SINT\22}", operator_name = "equal_width_shader"} : (tensor<4xi16>) -> tensor<4xi16>
  %out = tosa.add %shader, %rescaled : (tensor<4xi16>, tensor<4xi16>) -> tensor<4xi16>
  return %out : tensor<4xi16>
}

// -----

func.func @narrowing_rescale_not_rematerialized(%arg0: tensor<4xi16> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @narrowing_rescale_not_rematerialized
  // CHECK: func.func @graph_partition_0(%{{[^:]+}}: tensor<4xi16>) -> tensor<4xi8>
  // CHECK: tosa.rescale
  // CHECK: vgf.segment @graph_partition_1
  // CHECK: func.func @graph_partition_2(%{{[^:]+}}: tensor<4xi8>, %{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK-NOT: tosa.rescale
  // CHECK: tosa.add
  %multiplier = "tosa.const"() {values = dense<[1]> : tensor<1xi32>} : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<[0]> : tensor<1xi8>} : () -> tensor<1xi8>
  %input_zp = "tosa.const"() {values = dense<[0]> : tensor<1xi16>} : () -> tensor<1xi16>
  %output_zp = "tosa.const"() {values = dense<[0]> : tensor<1xi8>} : () -> tensor<1xi8>
  %rescaled = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %output_zp {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<4xi16>, tensor<1xi32>, tensor<1xi8>, tensor<1xi16>, tensor<1xi8>) -> tensor<4xi8>
  %shader = tosa.custom %arg0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R16_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "narrowing_shader"} : (tensor<4xi16>) -> tensor<4xi8>
  %out = tosa.add %shader, %rescaled : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  return %out : tensor<4xi8>
}
