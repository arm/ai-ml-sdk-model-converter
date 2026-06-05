//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: model-converter-opt --model-partition-marking --model-partitioning %s | FileCheck %s

func.func @first(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @first
  // CHECK: vgf.segment @graph_partition_0(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[FIRST_GRAPH0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.segment @graph_partition_1(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[FIRST_COMPUTE1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%[[FIRST_GRAPH0]]) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "first_shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[FIRST_COMPUTE1]] : tensor<4xi8>
  return %1 : tensor<4xi8>
}

func.func @second(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @second
  // CHECK: vgf.segment @graph_partition_0(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK-NOT: vgf.segment @graph_partition_1
  // CHECK: %[[SECOND_GRAPH0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[SECOND_GRAPH0]] : tensor<4xi8>
  return %0 : tensor<4xi8>
}

func.func @third(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @third
  // CHECK: %[[THIRD_GRAPH0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[THIRD_COMPUTE1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%[[THIRD_GRAPH0]]) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "third_shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[THIRD_GRAPH2:[0-9]+]] = vgf.run_segment @graph_partition_2 : (%[[THIRD_COMPUTE1]]) (tensor<4xi8>) -> tensor<4xi8>
  %2 = tosa.abs %1 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[THIRD_GRAPH2]] : tensor<4xi8>
  return %2 : tensor<4xi8>
}

func.func @fourth(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @fourth
  // CHECK: %[[FOURTH_COMPUTE0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.custom %arg0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "fourth_shader0"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[FOURTH_GRAPH1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%[[FOURTH_COMPUTE0]]) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.abs %0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[FOURTH_COMPUTE2:[0-9]+]] = vgf.run_segment @graph_partition_2 : (%[[FOURTH_GRAPH1]]) (tensor<4xi8>) -> tensor<4xi8>
  %2 = tosa.custom %1 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "fourth_shader1"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[FOURTH_COMPUTE2]] : tensor<4xi8>
  return %2 : tensor<4xi8>
}
