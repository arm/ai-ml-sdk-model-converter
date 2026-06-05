//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: model-converter-opt --split-input-file --model-partition-marking --model-partitioning %s | FileCheck %s
// RUN: model-converter-opt --split-input-file --model-partition-marking --model-partitioning %s | FileCheck %s --check-prefix=CLEANUP

// CLEANUP: module
// CLEANUP-NOT: graph_partition_delete
// CLEANUP-NOT: graph_partition_id
// CLEANUP-NOT: graph_partition_leaf_node
// CLEANUP-NOT: graph_partition_sequence_output_indices

func.func @graph_then_compute_segments(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @graph_then_compute_segments
  // CHECK: vgf.segment @graph_partition_0(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[GTC_G0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.segment @graph_partition_1(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[GTC_C1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%[[GTC_G0]]) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "gtc_shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[GTC_C1]] : tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

func.func @compute_then_graph_segments(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @compute_then_graph_segments
  // CHECK: vgf.segment @graph_partition_0(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[CTG_C0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.custom %arg0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "ctg_shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.segment @graph_partition_1(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[CTG_G1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%[[CTG_C0]]) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.abs %0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[CTG_G1]] : tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

func.func @compute_graph_compute_segments(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @compute_graph_compute_segments
  // CHECK: %[[CGC_C0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.custom %arg0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "cgc_shader0"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[CGC_G1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%[[CGC_C0]]) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.abs %0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[CGC_C2:[0-9]+]] = vgf.run_segment @graph_partition_2 : (%[[CGC_G1]]) (tensor<4xi8>) -> tensor<4xi8>
  %2 = tosa.custom %1 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "cgc_shader1"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[CGC_C2]] : tensor<4xi8>
  return %2 : tensor<4xi8>
}

// -----

func.func @compute_to_compute_segments(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @compute_to_compute_segments
  // CHECK: %[[CC_C0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.custom %arg0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "cc_shader0"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[CC_C1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%[[CC_C0]]) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "cc_shader1"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[CC_C1]] : tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

func.func @single_compute_segment(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @single_compute_segment
  // CHECK: vgf.segment @graph_partition_0(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK-SAME: segment_type = 1 : i32
  // CHECK-NEXT: %[[SC_PLACEHOLDER:[0-9]+]] = vgf.shader_placeholder
  // CHECK-NEXT: vgf.segment_output %[[SC_PLACEHOLDER]] : tensor<4xi8>
  // CHECK-NEXT: }
  // CHECK: %[[SC_C0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.custom %arg0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "single_compute_shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[SC_C0]] : tensor<4xi8>
  return %0 : tensor<4xi8>
}

// -----

func.func @compute_graph_compute_graph_compute_segments(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @compute_graph_compute_graph_compute_segments
  // CHECK: %[[CGCGC_C0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.custom %arg0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "cgcgc_shader0"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[CGCGC_G1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%[[CGCGC_C0]]) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.abs %0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[CGCGC_C2:[0-9]+]] = vgf.run_segment @graph_partition_2 : (%[[CGCGC_G1]]) (tensor<4xi8>) -> tensor<4xi8>
  %2 = tosa.custom %1 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "cgcgc_shader1"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[CGCGC_G3:[0-9]+]] = vgf.run_segment @graph_partition_3 : (%[[CGCGC_C2]]) (tensor<4xi8>) -> tensor<4xi8>
  %3 = tosa.abs %2 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[CGCGC_C4:[0-9]+]] = vgf.run_segment @graph_partition_4 : (%[[CGCGC_G3]]) (tensor<4xi8>) -> tensor<4xi8>
  %4 = tosa.custom %3 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "cgcgc_shader2"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[CGCGC_C4]] : tensor<4xi8>
  return %4 : tensor<4xi8>
}

// -----

func.func @duplicate_runtime_operand_deduplicates_segment_input(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @duplicate_runtime_operand_deduplicates_segment_input
  // CHECK: %[[DUP_RUNTIME_C0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.custom %arg0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "duplicate_runtime_shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.segment @graph_partition_1(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[DUP_RUNTIME_G1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%[[DUP_RUNTIME_C0]]) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.add %0, %0 : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[DUP_RUNTIME_G1]] : tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

func.func @duplicate_partition_outputs(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}, tensor<4xi8> {tf_saved_model.index_path = ["output_1"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0,output_1"}} {
  // CHECK-LABEL: vgf.sequence @duplicate_partition_outputs
  // CHECK: %[[DUP_G0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[DUP_G0]], %[[DUP_G0]] : tensor<4xi8>, tensor<4xi8>
  return %0, %0 : tensor<4xi8>, tensor<4xi8>
}

// -----

func.func @duplicate_passthrough_outputs(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}, tensor<4xi8> {tf_saved_model.index_path = ["output_1"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0,output_1"}} {
  // CHECK-LABEL: vgf.sequence @duplicate_passthrough_outputs
  // CHECK: vgf.segment @graph_partition_0(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[DUP_PASS:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  // CHECK: vgf.sequence_output %[[DUP_PASS]], %[[DUP_PASS]] : tensor<4xi8>, tensor<4xi8>
  return %arg0, %arg0 : tensor<4xi8>, tensor<4xi8>
}

// -----

func.func @distinct_passthrough_outputs_reordered(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}, %arg1: tensor<4xi8> {tf_saved_model.index_path = ["input_1"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}, tensor<4xi8> {tf_saved_model.index_path = ["output_1"]}) attributes {tf.entry_function = {inputs = "input_0,input_1", outputs = "output_0,output_1"}} {
  // CHECK-LABEL: vgf.sequence @distinct_passthrough_outputs_reordered
  // CHECK: vgf.segment @graph_partition_0(%{{[^:]+}}: tensor<4xi8>, %{{[^:]+}}: tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>)
  // CHECK: func.func @graph_partition_0(%[[DISTINCT_PASS0:[^:]+]]: tensor<4xi8>, %[[DISTINCT_PASS1:[^:]+]]: tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>)
  // CHECK: return %[[DISTINCT_PASS0]], %[[DISTINCT_PASS1]] : tensor<4xi8>, tensor<4xi8>
  // CHECK: %[[DISTINCT_PASS:[0-9]+]]:2 = vgf.run_segment @graph_partition_0 : (%arg1, %arg0) (tensor<4xi8>, tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>)
  // CHECK: vgf.sequence_output %[[DISTINCT_PASS]]#0, %[[DISTINCT_PASS]]#1 : tensor<4xi8>, tensor<4xi8>
  return %arg1, %arg0 : tensor<4xi8>, tensor<4xi8>
}

// -----

func.func @same_graph_partition_outputs_reordered(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}, tensor<4xi8> {tf_saved_model.index_path = ["output_1"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0,output_1"}} {
  // CHECK-LABEL: vgf.sequence @same_graph_partition_outputs_reordered
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[REORDERED:[0-9]+]]:2 = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>)
  %1 = tosa.add %0, %arg0 : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[REORDERED]]#0, %[[REORDERED]]#1 : tensor<4xi8>, tensor<4xi8>
  return %1, %0 : tensor<4xi8>, tensor<4xi8>
}

// -----

func.func @same_op_multiple_outputs_keep_result_order(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}, tensor<4xi8> {tf_saved_model.index_path = ["output_1"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0,output_1"}} {
  // CHECK-LABEL: vgf.sequence @same_op_multiple_outputs_keep_result_order
  // CHECK: %[[SAME_OP:[0-9]+]]:2 = tosa.custom
  // CHECK: return %[[SAME_OP]]#0, %[[SAME_OP]]#1 : tensor<4xi8>, tensor<4xi8>
  // CHECK: %[[SAME_OP_SEG:[0-9]+]]:2 = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>)
  // CHECK: vgf.sequence_output %[[SAME_OP_SEG]]#0, %[[SAME_OP_SEG]]#1 : tensor<4xi8>, tensor<4xi8>
  %0:2 = tosa.custom %arg0 {domain_name = "com.example.GraphCustom", implementation_attrs = "{}", operator_name = "graph_custom"} : (tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>)

  return %0#0, %0#1 : tensor<4xi8>, tensor<4xi8>
}

// -----

func.func @same_op_multiple_outputs_reordered(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}, tensor<4xi8> {tf_saved_model.index_path = ["output_1"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0,output_1"}} {
  // CHECK-LABEL: vgf.sequence @same_op_multiple_outputs_reordered
  // CHECK: %[[SAME_OP_REORDERED:[0-9]+]]:2 = tosa.custom
  // CHECK: return %[[SAME_OP_REORDERED]]#1, %[[SAME_OP_REORDERED]]#0 : tensor<4xi8>, tensor<4xi8>
  // CHECK: %[[SAME_OP_REORDERED_SEG:[0-9]+]]:2 = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>)
  // CHECK: vgf.sequence_output %[[SAME_OP_REORDERED_SEG]]#0, %[[SAME_OP_REORDERED_SEG]]#1 : tensor<4xi8>, tensor<4xi8>
  %0:2 = tosa.custom %arg0 {domain_name = "com.example.GraphCustom", implementation_attrs = "{}", operator_name = "graph_custom"} : (tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>)

  return %0#1, %0#0 : tensor<4xi8>, tensor<4xi8>
}

// -----

func.func @fft2d_multiple_outputs_keep_result_order() -> (tensor<1x32x32xf32> {tf_saved_model.index_path = ["output_0"]}, tensor<1x32x32xf32> {tf_saved_model.index_path = ["output_1"]}) attributes {tf.entry_function = {outputs = "output_0,output_1"}} {
  // CHECK-LABEL: vgf.sequence @fft2d_multiple_outputs_keep_result_order
  // CHECK: func.func @graph_partition_0() -> (tensor<1x32x32xf32>, tensor<1x32x32xf32>)
  // CHECK: %[[FFT_REAL:.*]], %[[FFT_IMAG:.*]] = tosa.fft2d
  // CHECK: return %[[FFT_REAL]], %[[FFT_IMAG]] : tensor<1x32x32xf32>, tensor<1x32x32xf32>
  // CHECK: %[[FFT_SEGMENT:[0-9]+]]:2 = vgf.run_segment @graph_partition_0 () -> (tensor<1x32x32xf32>, tensor<1x32x32xf32>)
  // CHECK: vgf.sequence_output %[[FFT_SEGMENT]]#0, %[[FFT_SEGMENT]]#1 : tensor<1x32x32xf32>, tensor<1x32x32xf32>
  %real = "tosa.const"() {values = dense<0.0> : tensor<1x32x32xf32>} : () -> tensor<1x32x32xf32>
  %imag = "tosa.const"() {values = dense<0.0> : tensor<1x32x32xf32>} : () -> tensor<1x32x32xf32>
  %output_real, %output_imag = tosa.fft2d %real, %imag {inverse = true} : (tensor<1x32x32xf32>, tensor<1x32x32xf32>) -> (tensor<1x32x32xf32>, tensor<1x32x32xf32>)

  return %output_real, %output_imag : tensor<1x32x32xf32>, tensor<1x32x32xf32>
}

// -----

func.func @rfft2d_multiple_outputs_keep_result_order() -> (tensor<1x32x17xf32> {tf_saved_model.index_path = ["output_0"]}, tensor<1x32x17xf32> {tf_saved_model.index_path = ["output_1"]}) attributes {tf.entry_function = {outputs = "output_0,output_1"}} {
  // CHECK-LABEL: vgf.sequence @rfft2d_multiple_outputs_keep_result_order
  // CHECK: func.func @graph_partition_0() -> (tensor<1x32x17xf32>, tensor<1x32x17xf32>)
  // CHECK: %[[RFFT_REAL:.*]], %[[RFFT_IMAG:.*]] = tosa.rfft2d
  // CHECK: return %[[RFFT_REAL]], %[[RFFT_IMAG]] : tensor<1x32x17xf32>, tensor<1x32x17xf32>
  // CHECK: %[[RFFT_SEGMENT:[0-9]+]]:2 = vgf.run_segment @graph_partition_0 () -> (tensor<1x32x17xf32>, tensor<1x32x17xf32>)
  // CHECK: vgf.sequence_output %[[RFFT_SEGMENT]]#0, %[[RFFT_SEGMENT]]#1 : tensor<1x32x17xf32>, tensor<1x32x17xf32>
  %input = "tosa.const"() {values = dense<0.0> : tensor<1x32x32xf32>} : () -> tensor<1x32x32xf32>
  %output_real, %output_imag = tosa.rfft2d %input : (tensor<1x32x32xf32>) -> (tensor<1x32x17xf32>, tensor<1x32x17xf32>)

  return %output_real, %output_imag : tensor<1x32x17xf32>, tensor<1x32x17xf32>
}

// -----

func.func @multi_result_op_exports_only_returned_result(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @multi_result_op_exports_only_returned_result
  // CHECK: %[[SINGLE_RETURN:[0-9]+]]:2 = tosa.custom
  // CHECK: return %[[SINGLE_RETURN]]#1 : tensor<4xi8>
  // CHECK: %[[SINGLE_RETURN_SEG:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  // CHECK: vgf.sequence_output %[[SINGLE_RETURN_SEG]] : tensor<4xi8>
  %0:2 = tosa.custom %arg0 {domain_name = "com.example.GraphCustom", implementation_attrs = "{}", operator_name = "graph_custom"} : (tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>)

  return %0#1 : tensor<4xi8>
}

// -----

func.func @multi_result_op_exports_only_cross_partition_result(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @multi_result_op_exports_only_cross_partition_result
  // CHECK: %[[CROSS:[0-9]+]]:2 = tosa.custom
  // CHECK: return %[[CROSS]]#1 : tensor<4xi8>
  // CHECK: %[[CROSS_G0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %0:2 = tosa.custom %arg0 {domain_name = "com.example.GraphCustom", implementation_attrs = "{}", operator_name = "graph_custom"} : (tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>)

  // CHECK: %[[CROSS_C1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%[[CROSS_G0]]) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.custom %0#1 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "cross_partition_shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[CROSS_C1]] : tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

func.func @multi_result_op_exports_returned_and_cross_partition_results(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}, tensor<4xi8> {tf_saved_model.index_path = ["output_1"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0,output_1"}} {
  // CHECK-LABEL: vgf.sequence @multi_result_op_exports_returned_and_cross_partition_results
  // CHECK: %[[MIXED:[0-9]+]]:2 = tosa.custom
  // CHECK: return %[[MIXED]]#0, %[[MIXED]]#1 : tensor<4xi8>, tensor<4xi8>
  // CHECK: %[[MIXED_G0:[0-9]+]]:2 = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>)
  %0:2 = tosa.custom %arg0 {domain_name = "com.example.GraphCustom", implementation_attrs = "{}", operator_name = "graph_custom"} : (tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>)

  // CHECK: %[[MIXED_C1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%[[MIXED_G0]]#1) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.custom %0#1 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "mixed_multi_result_shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[MIXED_G0]]#0, %[[MIXED_C1]] : tensor<4xi8>, tensor<4xi8>
  return %0#0, %1 : tensor<4xi8>, tensor<4xi8>
}

// -----

func.func @direct_constant_return() -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @direct_constant_return
  // CHECK: vgf.segment @graph_partition_0() -> tensor<4xi8>
  // CHECK: %[[DIRECT_CONST:[0-9]+]] = vgf.run_segment @graph_partition_0 () -> tensor<4xi8>
  %c0 = "tosa.const"() {values = dense<1> : tensor<4xi8>} : () -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[DIRECT_CONST]] : tensor<4xi8>
  return %c0 : tensor<4xi8>
}

// -----

func.func @mixed_passthrough_and_computed_outputs(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}, %arg1: tensor<4xi8> {tf_saved_model.index_path = ["input_1"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}, tensor<4xi8> {tf_saved_model.index_path = ["output_1"]}) attributes {tf.entry_function = {inputs = "input_0,input_1", outputs = "output_0,output_1"}} {
  // CHECK-LABEL: vgf.sequence @mixed_passthrough_and_computed_outputs
  // CHECK: vgf.segment @graph_partition_0(%{{[^:]+}}: tensor<4xi8>, %{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[MIXED_G0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0, %arg1) (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %arg1, %[[MIXED_G0]] : tensor<4xi8>, tensor<4xi8>
  return %arg1, %0 : tensor<4xi8>, tensor<4xi8>
}

// -----

func.func @constant_graph_first_then_compute_abi() -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @constant_graph_first_then_compute_abi
  // CHECK: vgf.segment @graph_partition_0() -> tensor<4xi8>
  // CHECK: %[[CGFC_CONST:[0-9]+]] = vgf.run_segment @graph_partition_0 () -> tensor<4xi8>
  %c0 = "tosa.const"() {values = dense<1> : tensor<4xi8>} : () -> tensor<4xi8>

  %0 = tosa.abs %c0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.segment @graph_partition_1(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[CGFC_C1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%[[CGFC_CONST]]) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.custom %c0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "cgfc_shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[CGFC_C1]] : tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

func.func @constant_compute_first_then_graph_abi() -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @constant_compute_first_then_graph_abi
  // CHECK: vgf.segment @graph_partition_0() -> tensor<4xi8>
  // CHECK: %[[CCFG_CONST:[0-9]+]] = vgf.run_segment @graph_partition_0 () -> tensor<4xi8>
  %c0 = "tosa.const"() {values = dense<1> : tensor<4xi8>} : () -> tensor<4xi8>

  // CHECK: %[[CCFG_C1:[0-9]+]] = vgf.run_segment @graph_partition_1 : (%[[CCFG_CONST]]) (tensor<4xi8>) -> tensor<4xi8>
  %0 = tosa.custom %c0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "ccfg_shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.segment @graph_partition_2(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK: %[[CCFG_G2:[0-9]+]] = vgf.run_segment @graph_partition_2 : (%[[CCFG_C1]]) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.add %0, %c0 : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[CCFG_G2]] : tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

func.func @non_vulkan_custom_remains_graph_segment(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<4xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // CHECK-LABEL: vgf.sequence @non_vulkan_custom_remains_graph_segment
  // CHECK: vgf.segment @graph_partition_0(%{{[^:]+}}: tensor<4xi8>) -> tensor<4xi8>
  // CHECK-SAME: segment_type = 0 : i32
  // CHECK: tosa.custom
  // CHECK-SAME: domain_name = "com.example.GraphCustom"
  // CHECK-NOT: vgf.shader_placeholder
  %0 = tosa.custom %arg0 {domain_name = "com.example.GraphCustom", implementation_attrs = "{}", operator_name = "graph_custom"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: %[[NV_G0:[0-9]+]] = vgf.run_segment @graph_partition_0 : (%arg0) (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.abs %0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: vgf.sequence_output %[[NV_G0]] : tensor<4xi8>
  return %1 : tensor<4xi8>
}
