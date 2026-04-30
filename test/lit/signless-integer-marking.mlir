//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: model-converter-opt --signless-integer-marking %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @unsigned_input(
  // CHECK-SAME: %{{.*}}: tensor<1x1xi8> {mlsdk.unsigned_input_output = true}
  // CHECK-SAME: ) -> (tensor<1x1xi16> {mlsdk.unsigned_input_output = false})
  func.func @unsigned_input(%arg0: tensor<1x1xi8>) -> tensor<1x1xi16> {
    %0 = "tosa.const"() <{values = dense<[8]> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = "tosa.const"() <{values = dense<[23]> : tensor<1xi8>}> : () -> tensor<1xi8>
    %2 = "tosa.const"() <{values = dense<[0]> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = "tosa.const"() <{values = dense<[0]> : tensor<1xi16>}> : () -> tensor<1xi16>
    %4 = tosa.rescale %arg0, %0, %1, %2, %3 {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = true, output_unsigned = false} : (tensor<1x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi16>) -> tensor<1x1xi16>
    return %4 : tensor<1x1xi16>
  }

  // CHECK-LABEL: func.func @signed_input(
  // CHECK-SAME: %{{.*}}: tensor<1x1xi8> {mlsdk.unsigned_input_output = false}
  // CHECK-SAME: ) -> (tensor<1x1xi16> {mlsdk.unsigned_input_output = false})
  func.func @signed_input(%arg0: tensor<1x1xi8>) -> tensor<1x1xi16> {
    %0 = "tosa.const"() <{values = dense<[8]> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = "tosa.const"() <{values = dense<[23]> : tensor<1xi8>}> : () -> tensor<1xi8>
    %2 = "tosa.const"() <{values = dense<[0]> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = "tosa.const"() <{values = dense<[0]> : tensor<1xi16>}> : () -> tensor<1xi16>
    %4 = tosa.rescale %arg0, %0, %1, %2, %3 {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<1x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi16>) -> tensor<1x1xi16>
    return %4 : tensor<1x1xi16>
  }

  // CHECK-LABEL: func.func @unsigned_output(
  // CHECK-SAME: %{{.*}}: tensor<1x1xi8> {mlsdk.unsigned_input_output = false}
  // CHECK-SAME: ) -> (tensor<1x1xi16> {mlsdk.unsigned_input_output = true})
  func.func @unsigned_output(%arg0: tensor<1x1xi8>) -> tensor<1x1xi16> {
    %0 = "tosa.const"() <{values = dense<[8]> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = "tosa.const"() <{values = dense<[23]> : tensor<1xi8>}> : () -> tensor<1xi8>
    %2 = "tosa.const"() <{values = dense<[0]> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = "tosa.const"() <{values = dense<[0]> : tensor<1xi16>}> : () -> tensor<1xi16>
    %4 = tosa.rescale %arg0, %0, %1, %2, %3 {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = true} : (tensor<1x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi16>) -> tensor<1x1xi16>
    return %4 : tensor<1x1xi16>
  }

  // CHECK-LABEL: func.func @const_input(
  // CHECK-SAME: %{{.*}}: tensor<1x1xi8> {mlsdk.unsigned_input_output = false}
  // CHECK-SAME: ) -> (tensor<1x1xi16> {mlsdk.unsigned_input_output = false})
  func.func @const_input(%arg0: tensor<1x1xi8>) -> tensor<1x1xi16> {
    %0 = "tosa.const"() <{values = dense<5> : tensor<1x1xi8>}> : () -> tensor<1x1xi8>
    %1 = "tosa.const"() <{values = dense<[8]> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = "tosa.const"() <{values = dense<[23]> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = "tosa.const"() <{values = dense<[0]> : tensor<1xi8>}> : () -> tensor<1xi8>
    %4 = "tosa.const"() <{values = dense<[0]> : tensor<1xi16>}> : () -> tensor<1xi16>
    %5 = tosa.rescale %0, %1, %2, %3, %4 {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = true, output_unsigned = false} : (tensor<1x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi16>) -> tensor<1x1xi16>
    return %5 : tensor<1x1xi16>
  }
}
