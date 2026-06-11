//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: model-converter-opt --split-input-file --model-partition-marking %s | FileCheck %s

// CHECK-LABEL: func.func @graph_only_chain
func.func @graph_only_chain(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: "tosa.const"
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = false
  %c0 = "tosa.const"() {values = dense<1> : tensor<4xi8>} : () -> tensor<4xi8>

  // CHECK: tosa.add
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = false
  %0 = tosa.add %arg0, %c0 : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %1 = tosa.abs %0 : (tensor<4xi8>) -> tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @multiple_sequence_outputs
func.func @multiple_sequence_outputs(%arg0: tensor<4xi8>, %arg1: tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>) {
  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 1>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %1 = tosa.abs %arg1 : (tensor<4xi8>) -> tensor<4xi8>
  return %1, %0 : tensor<4xi8>, tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @internal_cross_partition_and_return_use
func.func @internal_cross_partition_and_return_use(%arg0: tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>, tensor<4xi8>) {
  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 2>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %1 = tosa.abs %0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 1>
  %2 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader"} : (tensor<4xi8>) -> tensor<4xi8>
  return %1, %2, %0 : tensor<4xi8>, tensor<4xi8>, tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @duplicate_return_operands
func.func @duplicate_return_operands(%arg0: tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>) {
  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>
  return %0, %0 : tensor<4xi8>, tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @graph_then_compute_return
func.func @graph_then_compute_return(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader"} : (tensor<4xi8>) -> tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @compute_then_graph_return
func.func @compute_then_graph_return(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.custom %arg0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %1 = tosa.abs %0 : (tensor<4xi8>) -> tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @compute_then_graph_then_compute_return
func.func @compute_then_graph_then_compute_return(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.custom %arg0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader0"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %1 = tosa.abs %0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %2 = tosa.custom %1 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader1"} : (tensor<4xi8>) -> tensor<4xi8>
  return %2 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @compute_graph_compute_graph_compute_return
func.func @compute_graph_compute_graph_compute_return(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.custom %arg0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader0"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %1 = tosa.abs %0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %2 = tosa.custom %1 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader1"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 3 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %3 = tosa.abs %2 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 4 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %4 = tosa.custom %3 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader2"} : (tensor<4xi8>) -> tensor<4xi8>
  return %4 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @two_compute_boundaries
func.func @two_compute_boundaries(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader0"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %2 = tosa.abs %1 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 3 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %3 = tosa.custom %2 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader1"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 4 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %4 = tosa.abs %3 : (tensor<4xi8>) -> tensor<4xi8>
  return %4 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @compute_to_compute
func.func @compute_to_compute(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.custom %arg0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader0"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader1"} : (tensor<4xi8>) -> tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @constant_used_after_compute
func.func @constant_used_after_compute(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: "tosa.const"
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = false
  %table = "tosa.const"() {values = dense<0> : tensor<256xi8>} : () -> tensor<256xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.table
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %2 = tosa.table %1, %table : (tensor<4xi8>, tensor<256xi8>) -> tensor<4xi8>
  return %2 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @shared_graph_constant_across_partitions
func.func @shared_graph_constant_across_partitions(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: "tosa.const"
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = false
  %table = "tosa.const"() {values = dense<1> : tensor<256xi8>} : () -> tensor<256xi8>

  // CHECK: tosa.table
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.table %arg0, %table : (tensor<4xi8>, tensor<256xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.table
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %2 = tosa.table %1, %table : (tensor<4xi8>, tensor<256xi8>) -> tensor<4xi8>
  return %2 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @constant_used_by_compute_only
func.func @constant_used_by_compute_only() -> tensor<4xi8> {
  // CHECK: "tosa.const"
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %c0 = "tosa.const"() {values = dense<1> : tensor<4xi8>} : () -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %0 = tosa.custom %c0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader"} : (tensor<4xi8>) -> tensor<4xi8>
  return %0 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @constant_used_by_graph_and_compute
func.func @constant_used_by_graph_and_compute() -> tensor<4xi8> {
  // CHECK: "tosa.const"
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %c0 = "tosa.const"() {values = dense<1> : tensor<4xi8>} : () -> tensor<4xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.abs %c0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %1 = tosa.custom %c0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.add
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %2 = tosa.add %0, %1 : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  return %2 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @const_shape_used_after_compute
func.func @const_shape_used_after_compute(%arg0: tensor<4xi8>) -> tensor<2x2xi8> {
  // CHECK: tosa.const_shape
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = false
  %shape = "tosa.const_shape"() {values = dense<[2, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.reshape
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %2 = tosa.reshape %1, %shape : (tensor<4xi8>, !tosa.shape<2>) -> tensor<2x2xi8>
  return %2 : tensor<2x2xi8>
}

// -----

// CHECK-LABEL: func.func @const_shape_used_in_multiple_graph_partitions
func.func @const_shape_used_in_multiple_graph_partitions(%arg0: tensor<4xi8>) -> tensor<2x2xi8> {
  // CHECK: tosa.const_shape
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = false
  %shape = "tosa.const_shape"() {values = dense<[2, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>

  // CHECK: tosa.reshape
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.reshape %arg0, %shape : (tensor<4xi8>, !tosa.shape<2>) -> tensor<2x2xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader"} : (tensor<2x2xi8>) -> tensor<2x2xi8>

  // CHECK: tosa.reshape
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %2 = tosa.reshape %1, %shape : (tensor<2x2xi8>, !tosa.shape<2>) -> tensor<2x2xi8>
  return %2 : tensor<2x2xi8>
}

// -----

// CHECK-LABEL: func.func @constant_used_by_graph_before_later_compute
func.func @constant_used_by_graph_before_later_compute() -> (tensor<4xi8>, tensor<4xi8>) {
  // CHECK: "tosa.const"
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %c0 = "tosa.const"() {values = dense<1> : tensor<4xi8>} : () -> tensor<4xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %0 = tosa.abs %c0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 1>
  %1 = tosa.custom %c0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader"} : (tensor<4xi8>) -> tensor<4xi8>
  return %0, %1 : tensor<4xi8>, tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @constant_used_by_compute_before_multiple_later_graphs
func.func @constant_used_by_compute_before_multiple_later_graphs() -> tensor<4xi8> {
  // CHECK: "tosa.const"
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %c0 = "tosa.const"() {values = dense<1> : tensor<4xi8>} : () -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.custom %c0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader0"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.add
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %1 = tosa.add %0, %c0 : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 3 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %2 = tosa.custom %1 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader1"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.add
  // CHECK-SAME: graph_partition_id = 4 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %3 = tosa.add %2, %c0 : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  return %3 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @constant_return_only
func.func @constant_return_only() -> tensor<4xi8> {
  // CHECK: "tosa.const"
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %c0 = "tosa.const"() {values = dense<7> : tensor<4xi8>} : () -> tensor<4xi8>
  return %c0 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @unused_constant_in_graph_function
func.func @unused_constant_in_graph_function(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: "tosa.const"
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = false
  %c0 = "tosa.const"() {values = dense<9> : tensor<4xi8>} : () -> tensor<4xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>
  return %0 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @constant_used_in_multiple_late_graph_partitions
func.func @constant_used_in_multiple_late_graph_partitions(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: "tosa.const"
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = false
  %table = "tosa.const"() {values = dense<0> : tensor<256xi8>} : () -> tensor<256xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader0"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.table
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %2 = tosa.table %1, %table : (tensor<4xi8>, tensor<256xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 3 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %3 = tosa.custom %2 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader1"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.table
  // CHECK-SAME: graph_partition_id = 4 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %4 = tosa.table %3, %table : (tensor<4xi8>, tensor<256xi8>) -> tensor<4xi8>
  return %4 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @constant_used_by_compute_before_later_graph
func.func @constant_used_by_compute_before_later_graph() -> tensor<4xi8> {
  // CHECK: "tosa.const"
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %c0 = "tosa.const"() {values = dense<1> : tensor<4xi8>} : () -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.custom %c0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = false
  %1 = tosa.abs %0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.add
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %2 = tosa.add %1, %c0 : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  return %2 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @constant_used_by_multiple_compute_partitions
func.func @constant_used_by_multiple_compute_partitions() -> tensor<4xi8> {
  // CHECK: "tosa.const"
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %c0 = "tosa.const"() {values = dense<1> : tensor<4xi8>} : () -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %0 = tosa.custom %c0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader0"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  %1 = tosa.custom %c0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader1"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.add
  // CHECK-SAME: graph_partition_id = 3 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %2 = tosa.add %0, %1 : (tensor<4xi8>, tensor<4xi8>) -> tensor<4xi8>
  return %2 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @multiple_outputs_across_partitions
func.func @multiple_outputs_across_partitions(%arg0: tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>, tensor<4xi8>) {
  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 2>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 2 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 1>
  %2 = tosa.abs %1 : (tensor<4xi8>) -> tensor<4xi8>
  return %1, %2, %0 : tensor<4xi8>, tensor<4xi8>, tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @multi_result_op_outputs_keep_result_order
func.func @multi_result_op_outputs_keep_result_order(%arg0: tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>) {
  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0, 1>
  %0:2 = tosa.custom %arg0 {domain_name = "com.example.GraphCustom", implementation_attrs = "{}", operator_name = "graph_custom"} : (tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>)

  return %0#0, %0#1 : tensor<4xi8>, tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @multi_result_op_outputs_keep_sequence_order
func.func @multi_result_op_outputs_keep_sequence_order(%arg0: tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>) {
  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 1, 0>
  %0:2 = tosa.custom %arg0 {domain_name = "com.example.GraphCustom", implementation_attrs = "{}", operator_name = "graph_custom"} : (tensor<4xi8>) -> (tensor<4xi8>, tensor<4xi8>)

  return %0#1, %0#0 : tensor<4xi8>, tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @non_vulkan_custom_is_graph
func.func @non_vulkan_custom_is_graph(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = false
  %0 = tosa.custom %arg0 {domain_name = "com.example.GraphCustom", implementation_attrs = "{}", operator_name = "graph_custom"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %1 = tosa.abs %0 : (tensor<4xi8>) -> tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @stale_partition_attrs_are_rewritten
func.func @stale_partition_attrs_are_rewritten(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-NOT: graph_partition_sequence_output_indices
  %0 = tosa.abs %arg0 {graph_partition_id = 42 : i32, graph_partition_leaf_node = false, graph_partition_sequence_output_indices = array<i64: 7>} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", graph_partition_id = 43 : i32, graph_partition_leaf_node = false, graph_partition_sequence_output_indices = array<i64: 8>, implementation_attrs = "{}", operator_name = "shader"} : (tensor<4xi8>) -> tensor<4xi8>
  return %1 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @partition_ids_reset_after_many_partitions
func.func @partition_ids_reset_after_many_partitions(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 1 : i32
  %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader0"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 2 : i32
  %2 = tosa.abs %1 : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 3 : i32
  %3 = tosa.custom %2 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader1"} : (tensor<4xi8>) -> tensor<4xi8>

  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 4 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %4 = tosa.abs %3 : (tensor<4xi8>) -> tensor<4xi8>
  return %4 : tensor<4xi8>
}

// CHECK-LABEL: func.func @graph_only_resets_after_many_partitions
func.func @graph_only_resets_after_many_partitions(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.abs
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>
  return %0 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @partition_ids_reset_first
func.func @partition_ids_reset_first(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %0 = tosa.custom %arg0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader0"} : (tensor<4xi8>) -> tensor<4xi8>
  return %0 : tensor<4xi8>
}

// CHECK-LABEL: func.func @partition_ids_reset_second
func.func @partition_ids_reset_second(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  // CHECK: tosa.custom
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: graph_partition_sequence_output_indices = array<i64: 0>
  %0 = tosa.custom %arg0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{}", operator_name = "shader1"} : (tensor<4xi8>) -> tensor<4xi8>
  return %0 : tensor<4xi8>
}
