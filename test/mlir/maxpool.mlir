//
// SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

module attributes {tfl.description = "TOCO Converted.", tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<1x16x16x16xi8>) -> tensor<1x8x8x16xi8> attributes {tf.entry_function = {inputs = "data/Placeholder", outputs = "pool0/max_pooling2d/MaxPool"}} {
    %0 = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>, nan_mode = "IGNORE"} : (tensor<1x16x16x16xi8>) -> tensor<1x8x8x16xi8>
    return %0 : tensor<1x8x8x16xi8>
  }
}
