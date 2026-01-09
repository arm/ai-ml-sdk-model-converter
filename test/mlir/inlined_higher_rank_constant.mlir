//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

module {
  func.func @conv2d(%arg0: tensor<1x3x3x1xi8>) -> tensor<1x3x3x1xi8> {
    %weight = "tosa.const"() {values = dense<[[[[0], [1], [2]], [[3], [4], [5]], [[6], [7], [8]]]]> : tensor<1x3x3x1xi8>} : ()-> tensor<1x3x3x1xi8>
    %bias = "tosa.const"() {values = dense<0> : tensor<1xi32>} : ()-> tensor<1xi32>
    %input_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : ()-> tensor<1xi8>
    %weight_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : ()-> tensor<1xi8>
    %0 = tosa.conv2d %arg0, %weight, %bias, %input_zp, %weight_zp {acc_type = i32, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<1x3x3x1xi8>, tensor<1x3x3x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x3x3x1xi32>
    %1 = tosa.cast %0 : (tensor<1x3x3x1xi32>) -> tensor<1x3x3x1xi8>
    return %1 : tensor<1x3x3x1xi8>
  }
}
