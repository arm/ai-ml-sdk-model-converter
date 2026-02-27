//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

module attributes {tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32} {
  func.func @fp8E4M3_test(%arg0: tensor<4xf8E4M3FN>) -> tensor<4xf8E4M3FN> {
    %out = tosa.reverse %arg0 { axis = 0 : i32 } : (tensor<4xf8E4M3FN>) -> tensor<4xf8E4M3FN>
    return %out : tensor<4xf8E4M3FN>
  }
}
