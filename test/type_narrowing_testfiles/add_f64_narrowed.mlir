//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

module {
  func.func @main(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
    %0 = tosa.add %arg0, %arg1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}
