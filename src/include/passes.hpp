/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "include/type_narrowing.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"
#include "vgf-dialect/VGFDialect.h"

#include <memory>

namespace mlsdk::model_converter {
class VGFBuilder;
} // namespace mlsdk::model_converter

using mlsdk::model_converter::VGFBuilder;

namespace mlir::model_converter_passes {

#define GEN_PASS_CLASSES
#define GEN_PASS_DECL

void registerCheckConstantSparsityPass();
void registerDenseResourceInlinerPass();
void registerModelPartitionMarkingPass();
void registerSignlessIntegerMarkingPass();
void registerTosaShapedVerificationPass();
void registerModelPartitioningPass();
void registerSerializeVGFPass();
void registerTypeNarrowingPass();
void registerVGFConstantsPass();

#include "passes.hpp.inc"

std::unique_ptr<Pass> createVGFConstantsPass(std::shared_ptr<VGFBuilder> VGFBuilder);
std::unique_ptr<Pass> createSerializeVGFPass(std::shared_ptr<VGFBuilder> VGFBuilder, std::string outputName,
                                             const SerializeVGFPassOptions &options);
} // namespace mlir::model_converter_passes
