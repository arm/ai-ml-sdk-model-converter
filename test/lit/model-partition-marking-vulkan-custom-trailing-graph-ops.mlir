//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: model-converter-opt --split-input-file --model-partition-marking %s | FileCheck %s

// CHECK-LABEL: func.func @trailing_const
func.func @trailing_const(%arg0: tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["input_0"]}, %arg1: tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["input_1"]}, %arg2: tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["input_2"]}) -> (tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0,input_1,input_2", outputs = "output_0"}, tf_saved_model.exported_names = ["serving_default"]} {
  // CHECK: tosa.add
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.add %arg0, %arg1 : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %1 = tosa.custom %0, %arg2 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[16,16,16],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R32_SFLOAT\22,\22input_1_binding\22:1,\22input_1_descriptorset\22:0,\22input_1_type\22:\22TENSOR\22,\22input_1_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_1_vkformat\22:\22VK_FORMAT_R32_SFLOAT\22,\22output_0_binding\22:2,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R32_SFLOAT\22}", operator_name = "test_trailing_const_shader"} : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  // CHECK: "tosa.const"
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = false
  %2 = "tosa.const"() {values = dense<1.000000e+00> : tensor<1x16x16x16xf32>} : () -> tensor<1x16x16x16xf32>
  // CHECK: tosa.add
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %3 = tosa.add %1, %2 : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  return %3 : tensor<1x16x16x16xf32>
}

// -----

// CHECK-LABEL: func.func @multiple_trailing_consts
func.func @multiple_trailing_consts(%arg0: tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["input_0"]}, %arg1: tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["input_1"]}, %arg2: tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["input_2"]}) -> (tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0,input_1,input_2", outputs = "output_0"}, tf_saved_model.exported_names = ["serving_default"]} {
  // CHECK: tosa.add
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.add %arg0, %arg1 : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %1 = tosa.custom %0, %arg2 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[16,16,16],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R32_SFLOAT\22,\22input_1_binding\22:1,\22input_1_descriptorset\22:0,\22input_1_type\22:\22TENSOR\22,\22input_1_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_1_vkformat\22:\22VK_FORMAT_R32_SFLOAT\22,\22output_0_binding\22:2,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R32_SFLOAT\22}", operator_name = "test_trailing_const_shader"} : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  // CHECK: "tosa.const"
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = false
  %2 = "tosa.const"() {values = dense<1.000000e+00> : tensor<1x16x16x16xf32>} : () -> tensor<1x16x16x16xf32>
  // CHECK: "tosa.const"
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = false
  %3 = "tosa.const"() {values = dense<2.000000e+00> : tensor<1x16x16x16xf32>} : () -> tensor<1x16x16x16xf32>
  // CHECK: "tosa.const"
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = false
  %4 = "tosa.const"() {values = dense<3.000000e+00> : tensor<1x16x16x16xf32>} : () -> tensor<1x16x16x16xf32>
  // CHECK: tosa.add
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = false
  %5 = tosa.add %1, %2 : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  // CHECK: tosa.add
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = false
  %6 = tosa.add %5, %3 : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  // CHECK: tosa.add
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %7 = tosa.add %6, %4 : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  return %7 : tensor<1x16x16x16xf32>
}

// -----

// CHECK-LABEL: func.func @trailing_arg_only_graph_op
func.func @trailing_arg_only_graph_op(%arg0: tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["input_0"]}, %arg1: tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["input_1"]}, %arg2: tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["input_2"]}) -> (tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0,input_1,input_2", outputs = "output_0"}, tf_saved_model.exported_names = ["serving_default"]} {
  // CHECK: tosa.add
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.add %arg0, %arg1 : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %1 = tosa.custom %0, %arg2 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[16,16,16],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R32_SFLOAT\22,\22input_1_binding\22:1,\22input_1_descriptorset\22:0,\22input_1_type\22:\22TENSOR\22,\22input_1_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_1_vkformat\22:\22VK_FORMAT_R32_SFLOAT\22,\22output_0_binding\22:2,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R32_SFLOAT\22}", operator_name = "test_trailing_arg_only_graph_shader"} : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = false
  %2 = tosa.abs %arg0 : (tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  // CHECK: tosa.add
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %3 = tosa.add %1, %2 : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  return %3 : tensor<1x16x16x16xf32>
}

// -----

// CHECK-LABEL: func.func @cross_partition_input_before_raising_operand
func.func @cross_partition_input_before_raising_operand(%arg0: tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["input_0"]}, %arg1: tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["input_1"]}) -> (tensor<1x16x16x16xf32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0,input_1", outputs = "output_0"}, tf_saved_model.exported_names = ["serving_default"]} {
  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.abs %arg0 : (tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %1 = tosa.custom %arg1, %arg1 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[16,16,16],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R32_SFLOAT\22,\22input_1_binding\22:1,\22input_1_descriptorset\22:0,\22input_1_type\22:\22TENSOR\22,\22input_1_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_1_vkformat\22:\22VK_FORMAT_R32_SFLOAT\22,\22output_0_binding\22:2,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R32_SFLOAT\22}", operator_name = "test_late_partition_raise_shader"} : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = false
  %2 = tosa.abs %1 : (tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  // CHECK: tosa.add
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %3 = tosa.add %0, %2 : (tensor<1x16x16x16xf32>, tensor<1x16x16x16xf32>) -> tensor<1x16x16x16xf32>
  return %3 : tensor<1x16x16x16xf32>
}
