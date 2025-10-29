//
// SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32} {
    func.func @main(%arg0: tensor<1x1xi32> {tf_saved_model.index_path = ["x"]}) -> (tensor<1x1xi16> {tf_saved_model.index_path = ["model_input"]}) attributes {tf.entry_function = {inputs = "serving_default_x:0", outputs = "StatefulPartitionedCall:0"}, tf_saved_model.exported_names = ["serving_default"]} {
        %0 = "tosa.const"() {values = dense<5> : tensor<1x1xi8>} : () -> tensor<1x1xi8>
        %1 = "tosa.const"() {values = dense<[8]> : tensor<1xi32>} : () -> tensor<1xi32>
        %2 = "tosa.const"() {values = dense<[23]> : tensor<1xi8>} : () -> tensor<1xi8>
        %3 = "tosa.const"() {values = dense<[0]> : tensor<1xi8>} : () -> tensor<1xi8>
        %4 = "tosa.const"() {values = dense<[0]> : tensor<1xi16>} : () -> tensor<1xi16>
        %5 = tosa.rescale %0, %1, %2, %3, %4 {rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = true, output_unsigned = false} : (tensor<1x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi16>) -> tensor<1x1xi16>
        return %5 : tensor<1x1xi16>
    }
}
