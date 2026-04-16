//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

module {
  func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    %0 = "tosa.const"() <{values = dense<4.225000e+01> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.cast %arg0 : (tensor<1xf32>) -> tensor<1xf32>
    %2 = tosa.add %1, %0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %3 = tosa.cast %2 : (tensor<1xf32>) -> tensor<1xf32>
    return %3 : tensor<1xf32>
  }
}
