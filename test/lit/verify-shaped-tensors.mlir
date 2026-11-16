//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// RUN: split-file %s %t
// RUN: model-converter-opt --tosa-shape-verif %t/static.mlir
// RUN: not model-converter-opt --tosa-shape-verif %t/dynamic.mlir 2>&1 | FileCheck %s --check-prefix=DYNAMIC
// RUN: not model-converter-opt --tosa-shape-verif %t/unranked.mlir 2>&1 | FileCheck %s --check-prefix=UNRANKED
// RUN: not model-converter-opt --tosa-shape-verif %t/missing-inputs.mlir 2>&1 | FileCheck %s --check-prefix=DYNAMIC
// RUN: not model-converter-opt --allow-unregistered-dialect --tosa-shape-verif %t/internal.mlir 2>&1 | FileCheck %s --check-prefix=DYNAMIC
// RUN: not model-converter-opt --tosa-shape-verif %t/return.mlir 2>&1 | FileCheck %s --check-prefix=UNRANKED
// RUN: not model-converter-opt --tosa-shape-verif %t/zero.mlir 2>&1 | FileCheck %s --check-prefix=DYNAMIC
// DYNAMIC: error: Tensor dimensions must be statically positive
// UNRANKED: error: Unranked tensor is not supported

//--- static.mlir
func.func @static(%arg: tensor<4xf32>) -> tensor<4xf32> {
  return %arg : tensor<4xf32>
}
// Input-name count is irrelevant to shape validation.
func.func @named(%arg: tensor<4xf32>) -> tensor<4xf32>
    attributes {tf.entry_function = {inputs = "a,b"}} {
  return %arg : tensor<4xf32>
}

//--- dynamic.mlir
func.func @dynamic(%arg: tensor<?xf32>) -> tensor<?xf32>
    attributes {tf.entry_function = {inputs = "a"}} {
  return %arg : tensor<?xf32>
}

//--- unranked.mlir
func.func @unranked(%arg: tensor<*xf32>) -> tensor<*xf32>
    attributes {tf.entry_function = {inputs = "a"}} {
  return %arg : tensor<*xf32>
}

//--- missing-inputs.mlir
func.func @missing(%arg: tensor<?xf32>) -> tensor<?xf32>
    attributes {tf.entry_function = {}} {
  return %arg : tensor<?xf32>
}

//--- internal.mlir
func.func @internal(%arg: tensor<4xf32>) -> tensor<4xf32>
    attributes {tf.entry_function = {inputs = "a"}} {
  %unused = "test.result"() : () -> tensor<?xf32>
  return %arg : tensor<4xf32>
}

//--- return.mlir
func.func private @declaration() -> tensor<*xf32>

//--- zero.mlir
func.func @zero(%arg: tensor<0xf32>) -> tensor<0xf32> {
  return %arg : tensor<0xf32>
}
