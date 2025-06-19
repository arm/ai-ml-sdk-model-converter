/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "include/passes.hpp"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;
using namespace mlir::model_converter_passes;

int main(int argc, char **argv) {
    DialectRegistry registry;
    registry.insert<func::FuncDialect, tosa::TosaDialect, spirv::SPIRVDialect, vgf::VGFDialect>();

    registerCheckConstantSparsityPass();
    registerDenseResourceInlinerPass();
    registerModelPartitionMarkingPass();
    registerSignlessIntegerMarkingPass();
    registerTosaShapedVerificationPass();
    registerModelPartitioningPass();
    registerSerializeVGFPass();
    registerTypeNarrowingPass();
    registerVGFConstantsPass();

    return asMainReturnCode(MlirOptMain(argc, argv, "ML SDK Model Converter MLIR optimizer driver\n", registry));
}
