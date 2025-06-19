/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef MLIR_CONVERSION_PASSES
#define MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def SerializeVGFPass : Pass<"serialize-vgf", "mlir::vgf::SequenceOp"> {
  let summary = "Serialize VGF MLIR to VGF binary";
  let description = [{Serialize the Module representing VGF MLIR to VGF binary.}];
  let options = [
    Option<"emitDebugInfo", "emitDebugInfo", "bool", "false","Emit debug information">,
  ];
}

def VGFConstantsPass : Pass<"vgf-constants", "mlir::ModuleOp"> {
  let summary = "Serialize TOSA Constants to VGF";
  let description = [{Serialize TOSA MLIR constants to VGF.}];
}

def TosaShapedVerificationPass : Pass<"tosa-shape-verif", "func::FuncOp"> {
  let summary = "Optional pass that asserts no tensors in the graph are unranked or have dynamic shapes.";
  let description = [{}];
}

def CheckConstantSparsityPass : Pass<"check-constant-sparsity", "mlir::ModuleOp"> {
  let summary = "Detect if a TOSA constant is sparse";
  let description = [{
    Detect if a TOSA constant is 2:4 sparse and mark as sparse in TOSA MLIR.
  }];
}

def ModelPartitionMarkingPass : Pass<"model-partition-marking", "mlir::ModuleOp"> {
  let summary = "Mark each operation in the model with a partition attribute";
  let description = [{
    Mark each operation in the model with a partition attribute that describe in which model partition the operation
    should belong to.  For example, tosa.custom operation representing shader should be placed in different partitions
    wrt regular tosa graph operation (e.g., conv2d).
  }];
}

def ModelPartitioningPass : Pass<"model-partitioning", "mlir::ModuleOp"> {
  let summary = "Split the model in several sub-modules one per partition";
  let description = [{
    Split the model in several modules one per partition following the marking done by the ModelPartitionMarkingPass.
  }];
  let options = [
      Option<"analysis", "analysis", "bool", "false","Analysis">
  ];
}

def SignlessIntegerMarkingPass : Pass<"signless-integer-marking", "mlir::func::FuncOp"> {
  let summary = "Mark signless integers as signed or unsigned";
  let description = [{}];
}

def TypeNarrowingPass : Pass<"type-narrowing", "mlir::ModuleOp"> {
  let summary = "Narrow types of operations' operands/results";
  let description = [{
    Reduce the size of the input floating point model by quantizing the weights to narrower floating-point types;
    e.g. float16. Moreover, reduce the operational precision by type-casting all the internal operations as well;
    something that can improve the overall execution time.
  }];
  let options = [
      Option<"mode", "mode", "TypeNarrowingMode", "TypeNarrowingMode::Full","Type narrowing",
            [{::llvm::cl::values(
               clEnumValN(TypeNarrowingMode::Full, "full", "Full type narrowing"),
               clEnumValN(TypeNarrowingMode::Partial, "partial", "Partial type narrowing"),
               clEnumValN(TypeNarrowingMode::FullPreserveIO, "full_preserve_io", "Full type narrowing with i/o types preserved")
            )}]>
  ];
}

def DenseResourceInlinerPass : Pass<"dense-resource-inliner", "func::FuncOp"> {
  let summary = "Inline DenseResourceElementsAttr into DenseElementsAttr";
  let description = [{}];
}

#endif // MLIR_CONVERSION_PASSES
