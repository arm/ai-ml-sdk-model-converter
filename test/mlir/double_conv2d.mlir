//
// SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

module attributes {tf_saved_model.semantics, tosa.description = "Tosa FBS Converted", tosa.fbs_version = "0.60.0"} {
    func.func @main(%arg0: tensor<1x32x32x23xi8> {tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x32x32x1xi32> {tf_saved_model.index_path = ["output_0"]}) attributes {tf.entry_function = {inputs = "tosa_deserialized_input_0:0", outputs = "tosa_deserialized_output_0:0"}, tf_saved_model.exported_names = ["tosa_deserialized"]} {
    %0 = "tosa.const"() {values = dense<"0x00000058005B0000000000E200006400760000000000000000006C5600000000000055750000005E000000000000000000380000000971000000000000AF000000B70000000000001C0000007167000000000067000000780000000000000040000000D3000000227D0000000000007E000000000000BE0000009C000000A300007300005200000000000078000000007A00000000850000007A000000FB00000061000000000000115D0000006A000000006D000000000000000057000000E5000000AA5400000000000034000000"> : tensor<1x3x3x23xi8>} : () -> tensor<1x3x3x23xi8>
    %1 = "tosa.const"() {values = dense<12> : tensor<1xi32>} : () -> tensor<1xi32>
    %2 = "tosa.const"() {values = dense<[[[[0, 0, 55, 88, 0], [91, 0, 88, 0, 0], [37, 0, 0, -52, 0]], [[0, 118, 0, -30, 0], [74, 79, 0, 0, 0], [0, 108, 86, 0, 0]], [[0, 37, 0, -34, 0], [117, 71, 0, 0, 0], [122, 0, 76, 0, 0]]]]> : tensor<1x3x3x5xi8>} : () -> tensor<1x3x3x5xi8>
    %3 = "tosa.const"() {values = dense<12> : tensor<1xi32>} : () -> tensor<1xi32>
    %4 = "tosa.const"() {values = dense<2026291432> : tensor<1xi32>} : () -> tensor<1xi32>
    %5 = "tosa.const"() {values = dense<40> : tensor<1xi8>} : () -> tensor<1xi8>
    %6 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %7 = tosa.conv2d %arg0, %0, %1, %6, %6 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, acc_type = i32} : (tensor<1x32x32x23xi8>, tensor<1x3x3x23xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x32x32x1xi32>
    %8 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %9 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %10 = tosa.rescale %7, %4, %5, %9, %8 {rounding_mode = DOUBLE_ROUND, per_channel = true, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<1x32x32x1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x32x32x1xi8>
    %11 = tosa.conv2d %10, %2, %3, %8, %8 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, acc_type = i32} : (tensor<1x32x32x1xi8>, tensor<1x3x3x5xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x32x32x1xi32>
    return %11 : tensor<1x32x32x1xi32>
    }
}
