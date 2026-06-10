//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: model-converter-opt --assign-grapharm-interface-var-abi %s | FileCheck %s

vgf.sequence @main(%arg0: tensor<1xi8>) -> tensor<1xi8> {
  vgf.segment @single_partition_0(%arg1: tensor<1xi8>) -> tensor<1xi8> attributes {segment_type = 0 : i32} {
    // CHECK-LABEL: func.func @single_partition_0
    // CHECK-SAME: grapharm.interface_var_abi = #spirv.interface_var_abi<(0, 0)>
    // CHECK-SAME: grapharm.interface_var_abi = #spirv.interface_var_abi<(0, 1)>
    func.func @single_partition_0(%arg2: tensor<1xi8>) -> tensor<1xi8> {
      return %arg2 : tensor<1xi8>
    }
    vgf.segment_output
  }
  %0 = vgf.run_segment @single_partition_0 :(%arg0) (tensor<1xi8>) -> tensor<1xi8>
  vgf.sequence_output %0 : tensor<1xi8>
}

vgf.sequence @multi_segment(%arg0: tensor<1xi8>) -> tensor<1xi8> {
  vgf.segment @multi_partition_0(%arg1: tensor<1xi8>) -> tensor<1xi8> attributes {segment_type = 0 : i32} {
    // CHECK-LABEL: func.func @multi_partition_0
    // CHECK-SAME: grapharm.interface_var_abi = #spirv.interface_var_abi<(0, 0)>
    // CHECK-SAME: grapharm.interface_var_abi = #spirv.interface_var_abi<(0, 1)>
    func.func @multi_partition_0(%arg2: tensor<1xi8>) -> tensor<1xi8> {
      return %arg2 : tensor<1xi8>
    }
    vgf.segment_output
  }
  %0 = vgf.run_segment @multi_partition_0 :(%arg0) (tensor<1xi8>) -> tensor<1xi8>
  vgf.segment @multi_partition_1(%arg1: tensor<1xi8>) -> tensor<1xi8> attributes {segment_type = 0 : i32} {
    // CHECK-LABEL: func.func @multi_partition_1
    // CHECK-SAME: grapharm.interface_var_abi = #spirv.interface_var_abi<(0, 1)>
    // CHECK-SAME: grapharm.interface_var_abi = #spirv.interface_var_abi<(0, 2)>
    func.func @multi_partition_1(%arg2: tensor<1xi8>) -> tensor<1xi8> {
      return %arg2 : tensor<1xi8>
    }
    vgf.segment_output
  }
  %1 = vgf.run_segment @multi_partition_1 :(%0) (tensor<1xi8>) -> tensor<1xi8>
  vgf.sequence_output %1 : tensor<1xi8>
}
