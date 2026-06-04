//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: model-converter-opt --model-partition-marking --model-partitioning %s | FileCheck %s

// CHECK-LABEL: vgf.sequence @main
// CHECK: vgf.segment @graph_partition_0
// CHECK: func.func @graph_partition_0(%{{.*}}: tensor<4xi8>) -> tensor<4xi8>
// CHECK: vgf.segment @graph_partition_1
// CHECK: vgf.segment @graph_partition_2
func.func @main(%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<2x2xi8> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "input_0", outputs = "output_0"}} {
  // The table is consumed after the partition boundary and should be cloned into
  // the consuming graph segment instead of becoming a runtime input.
  // CHECK: func.func @graph_partition_2(%{{.*}}: tensor<4xi8>) -> tensor<2x2xi8>
  // CHECK-NOT: tensor<256xi8>
  // CHECK: "tosa.const"() <{values = dense<0> : tensor<256xi8>}> {graph_partition_id = 0 : i32, graph_partition_leaf_node = true, spirv_graph_constant_id = 0 : i32} : () -> tensor<256xi8>
  %table = "tosa.const"() {spirv_graph_constant_id = 0 : i32, values = dense<0> : tensor<256xi8>} : () -> tensor<256xi8>

  // Shape constants are also compile-time constants and should be cloned rather
  // than added as graph segment arguments.
  // CHECK-NOT: !tosa.shape<2>
  // CHECK: tosa.const_shape
  // CHECK-SAME: graph_partition_id = 0 : i32
  // CHECK-SAME: graph_partition_leaf_node = true
  // CHECK-SAME: values = dense<2> : tensor<2xindex>
  // CHECK: tosa.table
  // CHECK: tosa.reshape
  %shape = "tosa.const_shape"() {values = dense<[2, 2]> : tensor<2xindex>} : () -> !tosa.shape<2>

  %0 = tosa.abs %arg0 : (tensor<4xi8>) -> tensor<4xi8>
  %1 = tosa.custom %0 {domain_name = "com.arm.VulkanCustomShader", implementation_attrs = "{\22entry_point\22:\22main\22,\22is_vkshader\22:true,\22workgroup_sizes\22:[1,1,1],\22input_0_binding\22:0,\22input_0_descriptorset\22:0,\22input_0_type\22:\22TENSOR\22,\22input_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22input_0_vkformat\22:\22VK_FORMAT_R8_SINT\22,\22output_0_binding\22:1,\22output_0_descriptorset\22:0,\22output_0_type\22:\22TENSOR\22,\22output_0_vkdescriptortype\22:\22VK_DESCRIPTOR_TYPE_TENSOR_ARM\22,\22output_0_vkformat\22:\22VK_FORMAT_R8_SINT\22}", operator_name = "partition_barrier"} : (tensor<4xi8>) -> tensor<4xi8>
  %2 = tosa.table %1, %table : (tensor<4xi8>, tensor<256xi8>) -> tensor<4xi8>
  %3 = tosa.reshape %2, %shape : (tensor<4xi8>, !tosa.shape<2>) -> tensor<2x2xi8>
  return %3 : tensor<2x2xi8>
}
