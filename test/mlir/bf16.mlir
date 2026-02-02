//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32} {
  func.func @bf16_test(%arg0: tensor<4xbf16>) -> tensor<4xbf16> {
    %cst = "tosa.const"() {
      values = dense<[0.0, 1.0, -2.0, 3.5]> : tensor<4xbf16>
    } : () -> tensor<4xbf16>

    %abs = "tosa.abs"(%arg0) : (tensor<4xbf16>) -> tensor<4xbf16>
    %out = "tosa.add"(%abs, %cst) : (tensor<4xbf16>, tensor<4xbf16>) -> tensor<4xbf16>
    return %out : tensor<4xbf16>
  }
}
