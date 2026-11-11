//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: split-file %s %t
// RUN: model-converter-opt --dense-resource-inliner %t/plain.mlir | FileCheck %s --check-prefix=PLAIN
// RUN: %if block-scaled-tensors %{ model-converter-opt --dense-resource-inliner %t/block-scaled.mlir | FileCheck %s --check-prefix=SCALED %}

//--- plain.mlir
// PLAIN-LABEL: func.func @plain_constant()
// PLAIN: tosa.const values(dense<[-7, 42]> : tensor<2xi32>) : () -> tensor<2xi32>
func.func @plain_constant() -> tensor<2xi32> {
  %0 = tosa.const values(dense_resource<plain_values> : tensor<2xi32>) : () -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

{-#
  dialect_resources: {
    builtin: {
      plain_values: "0x04000000F9FFFFFF2A000000"
    }
  }
#-}

//--- block-scaled.mlir
// Two blocks with different payloads and non-unit scales. Their decoded
// values are 0.75 and -10.0, respectively; inlining must preserve the stored
// values (1.5 and -2.5) and scales (0.5 and 4.0) separately.
// SCALED-LABEL: func.func @block_scaled_constant()
// SCALED: tosa.const values(dense<tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN, {5.000000e-01, 4.000000e+00}>> :
// SCALED-SAME: {{\[\[(1\.500000e\+00 : f8E4M3FN, ){31}1\.500000e\+00 : f8E4M3FN\],}}
// SCALED-SAME: {{\[(-2\.500000e\+00 : f8E4M3FN, ){31}-2\.500000e\+00 : f8E4M3FN\]\]}}
// SCALED-SAME: >) : () -> tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
func.func @block_scaled_constant() -> tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>> {
  %0 = tosa.const values(dense_resource<scaled_values> : tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN, {0.5, 4.0}>>) : () -> tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
  return %0 : tensor<2x32x!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:f8E4M3FN>>
}

{-#
  dialect_resources: {
    builtin: {
      // Alignment 1 encoded in four bytes, then 32 FP8 E4M3FN values of
      // 1.5 (0x3c) followed by 32 values of -2.5 (0xc2).
      scaled_values: "0x010000003C3C3C3C3C3C3C3C3C3C3C3C3C3C3C3C3C3C3C3C3C3C3C3C3C3C3C3C3C3C3C3CC2C2C2C2C2C2C2C2C2C2C2C2C2C2C2C2C2C2C2C2C2C2C2C2C2C2C2C2C2C2C2C2"
    }
  }
#-}
