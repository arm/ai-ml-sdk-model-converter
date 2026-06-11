//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: model-converter-opt --split-input-file --model-partition-marking --model-partitioning %s | FileCheck %s

func.func @passthrough_function_creates_graph_segment(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @passthrough_function_creates_graph_segment
  // CHECK: vgf.segment @graph_partition_0(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[PASSTHROUGH:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  // CHECK: vgf.sequence_output %[[PASSTHROUGH]] : tensor<4xi8>
  return %arg0 : tensor<4xi8>
}

// -----

func.func @single_graph_partition_keeps_unused_args(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}, %arg1: tensor<4xi8> {tf_saved_model.index_path = ["input_1"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0,input_1", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @single_graph_partition_keeps_unused_args
  // CHECK: vgf.segment @graph_partition_0(%{{[^:]+}}: tensor<4xi8>, %{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[SINGLE_GRAPH:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0, %arg1) (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[SINGLE_GRAPH]] : tensor<4xi8>
  return %0 : tensor<4xi8>
}

// -----

func.func @early_graph_output_used_by_later_graph(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}, %arg1: tensor<4xi8> {tf_saved_model.index_path = ["input_1"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0,input_1", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @early_graph_output_used_by_later_graph
  // CHECK: vgf.segment @graph_partition_0(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[GRAPH0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.segment @graph_partition_1(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[COMPUTE1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%arg1) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.custom %arg1 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "unrelated_shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.segment @graph_partition_2(%{{[^:]+}}: tensor<4xi8>, %{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  %2 = tosa.abs %1 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[GRAPH2:[0-9]+]] = vgf.run_segment @graph_partition_2 : (%[[COMPUTE1]], %[[GRAPH0]]) (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  %3 = tosa.add %0, %2 : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[GRAPH2]] : tensor<4xi8>
  return %3 : tensor<4xi8>
}

// -----

func.func @early_graph_output_used_by_later_compute(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}, %arg1: tensor<4xi8> {tf_saved_model.index_path = ["input_1"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}, tensor<4xi8> {tf_saved_model.index_path = ["output_1"]}) attributes {tf.entry_function = {inputs = "input_0,input_1", outputs = "output_0,output_1"}} {
  // CHECK-LABEL: vgf.sequence @early_graph_output_used_by_later_compute
  // CHECK: vgf.segment @graph_partition_0(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[GRAPH0B:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.segment @graph_partition_1(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[COMPUTE1B:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%arg1) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.custom %arg1 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "first_shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.segment @graph_partition_2(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[GRAPH2B:[0-9]+]] = vgf.run_segment @graph_partition_2 : (%[[COMPUTE1B]]) (tensor<4xi8>) -> tensor<4xi8>
  %2 = tosa.abs %1 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.segment @graph_partition_3(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[COMPUTE3B:[0-9]+]] = vgf.run_segment @graph_partition_3 : (%[[GRAPH0B]]) (tensor<4xi8>) -> tensor<4xi8>
  %3 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "late_shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[GRAPH2B]], %[[COMPUTE3B]] : tensor<4xi8>, tensor<4xi8>
  return %2, %3 : tensor<4xi8>, tensor<4xi8>
}

// -----

func.func @runtime_producer_fanout_to_later_partitions(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @runtime_producer_fanout_to_later_partitions
  // CHECK: vgf.segment @graph_partition_0(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[FANOUT_GRAPH0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.segment @graph_partition_1(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[FANOUT_COMPUTE1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%[[FANOUT_GRAPH0]]) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "fanout_shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.segment @graph_partition_2(%{{[^:]+}}: tensor<4xi8>, %{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[FANOUT_GRAPH2:[0-9]+]] = vgf.run_segment @graph_partition_2 : (%[[FANOUT_COMPUTE1]], %[[FANOUT_GRAPH0]]) (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  %2 = tosa.add %1, %0 : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[FANOUT_GRAPH2]] : tensor<4xi8>
  return %2 : tensor<4xi8>
}

// -----

func.func @later_partition_joins_multiple_non_adjacent_inputs(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}, %arg1: tensor<4xi8> {tf_saved_model.index_path = ["input_1"]}, %arg2: tensor<4xi8> {tf_saved_model.index_path = ["input_2"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0,input_1,input_2", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @later_partition_joins_multiple_non_adjacent_inputs
  // CHECK: %[[JOIN_GRAPH0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[JOIN_COMPUTE1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%arg1) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.custom %arg1 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "join_shader0"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[JOIN_GRAPH2:[0-9]+]] = vgf.run_segment @graph_partition_2 : (%[[JOIN_COMPUTE1]]) (tensor<4xi8>) -> tensor<4xi8>
  %2 = tosa.abs %1 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[JOIN_COMPUTE3:[0-9]+]] = vgf.run_segment @graph_partition_3 : (%arg2) (tensor<4xi8>) -> tensor<4xi8>
  %3 = tosa.custom %arg2 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "join_shader1"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.segment @graph_partition_4(%{{[^:]+}}: tensor<4xi8>, %{{[^:]+}}: tensor<4xi8>, %{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  %4 = tosa.add %0, %3 : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[JOIN_GRAPH4:[0-9]+]] = vgf.run_segment @graph_partition_4 : (%[[JOIN_GRAPH0]], %[[JOIN_COMPUTE3]], %[[JOIN_GRAPH2]]) (tensor<4xi8>, tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  %5 = tosa.add %4, %2 : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[JOIN_GRAPH4]] : tensor<4xi8>
  return %5 : tensor<4xi8>
}

// -----

func.func @partitioned_multiple_outputs_keep_sequence_order(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}, tensor<4xi8> {tf_saved_model.index_path = ["output_1"]}, tensor<4xi8> {tf_saved_model.index_path = ["output_2"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0,output_1,output_2"}} {
  // CHECK-LABEL: vgf.sequence @partitioned_multiple_outputs_keep_sequence_order
  // CHECK: %[[MULTI_GRAPH0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[MULTI_COMPUTE1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%[[MULTI_GRAPH0]]) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "multi_output_shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[MULTI_GRAPH2:[0-9]+]] = vgf.run_segment @graph_partition_2 : (%[[MULTI_COMPUTE1]]) (tensor<4xi8>) -> tensor<4xi8>
  %2 = tosa.abs %1 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[MULTI_COMPUTE1]], %[[MULTI_GRAPH2]], %[[MULTI_GRAPH0]] : tensor<4xi8>, tensor<4xi8>, tensor<4xi8>
  return %1, %2, %0 : tensor<4xi8>, tensor<4xi8>, tensor<4xi8>
}

// -----

func.func @same_value_used_by_later_graph_and_compute(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}, %arg1: tensor<4xi8> {tf_saved_model.index_path = ["input_1"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}, tensor<4xi8> {tf_saved_model.index_path = ["output_1"]}) attributes {tf.entry_function = {inputs = "input_0,input_1", outputs = "output_0,output_1"}} {
  // CHECK-LABEL: vgf.sequence @same_value_used_by_later_graph_and_compute
  // CHECK: %[[SHARED_GRAPH0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[SHARED_COMPUTE1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%arg1) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.custom %arg1 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "shared_shader0"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.segment @graph_partition_2(%{{[^:]+}}: tensor<4xi8>, %{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[SHARED_GRAPH2:[0-9]+]] = vgf.run_segment @graph_partition_2 : (%[[SHARED_COMPUTE1]], %[[SHARED_GRAPH0]]) (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  %2 = tosa.add %1, %0 : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.segment @graph_partition_3(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[SHARED_COMPUTE3:[0-9]+]] = vgf.run_segment @graph_partition_3 : (%[[SHARED_GRAPH0]]) (tensor<4xi8>) -> tensor<4xi8>
  %3 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "shared_shader1"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[SHARED_GRAPH2]], %[[SHARED_COMPUTE3]] : tensor<4xi8>, tensor<4xi8>
  return %2, %3 : tensor<4xi8>, tensor<4xi8>
}

// -----

func.func @constant_used_by_multiple_compute_partitions_abi() -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}, tensor<4xi8> {tf_saved_model.index_path = ["output_1"]}) attributes {tf.entry_function = {inputs = "", outputs = "output_0,output_1"}} {
  // CHECK-LABEL: vgf.sequence @constant_used_by_multiple_compute_partitions_abi
  // CHECK: vgf.segment @graph_partition_0() -> tensor<4xi8>
  // CHECK: %[[CONST_GRAPH0:[0-9]+]] = vgf.run_segment @graph_partition_0 () -> tensor<4xi8>
  %c0 = "tosa.const"() {values = dense<1> : tensor<4xi8>} : () -> tensor<4xi8>

  // CHECK: vgf.segment @graph_partition_1(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[CONST_COMPUTE1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%[[CONST_GRAPH0]]) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.custom %c0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "const_shader0"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.segment @graph_partition_2(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[CONST_COMPUTE2:[0-9]+]] = vgf.run_segment @graph_partition_2 : (%[[CONST_GRAPH0]]) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.custom %c0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "const_shader1"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[CONST_COMPUTE1]], %[[CONST_COMPUTE2]] : tensor<4xi8>, tensor<4xi8>
  return %0, %1 : tensor<4xi8>, tensor<4xi8>
}
