/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "compiler.hpp"
#include "include/passes.hpp"
#include "mlir/Conversion/TosaToSPIRV/ConvertTosaConstants.h"
#include "mlir/Conversion/TosaToSPIRV/TosaToSPIRV.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "vgf-dialect/VGFDialect.h"
#include "vgf_builder.hpp"

#include "include/DeserializationPasses.h" // from @tosa_mlir_translator
#include "include/SerializationPasses.h"   // from @tosa_mlir_translator

#include <filesystem>
#include <iostream>

extern llvm::cl::opt<std::string> tosa_flatbuffer_filename;
extern llvm::cl::opt<std::string> tosa_flatbuffer_schema;

using namespace mlir::model_converter_passes;

namespace mlsdk::model_converter {

namespace {
bool isTosaFlatbuffer(const std::string &input) { return std::filesystem::path(input).extension() == ".tosa"; }
} // namespace

Compiler::Compiler(const Options &options)
    : _pm(&_context, "builtin.module"), _sourceMgrHandler(_sourceMgr, &_context), _options(options) {}

void Compiler::SetRegistry() {
    _registry.insert<mlir::func::FuncDialect>();
    _registry.insert<mlir::tosa::TosaDialect>();
    _registry.insert<mlir::spirv::SPIRVDialect>();
    _registry.insert<vgf::VGFDialect>();
    _context.appendDialectRegistry(_registry);
    _context.loadAllAvailableDialects();
}

void Compiler::SetMultiThreading(bool enable) { _context.enableMultithreading(enable); }

void Compiler::SetLogging() {
    _context.printOpOnDiagnostic(_options.analysis);
    _context.printStackTraceOnDiagnostic(false);
}

void Compiler::SetPassManager() {
    _pm.enableVerifier(_options.enable_verifier);
    if (_options.enable_statistics) {
        _pm.enableStatistics();
    }
    if (_options.dump_mlir) {
        _pm.enableIRPrinting();
    }

    if (!_options.emit_debug_info) {
        _pm.addPass(createStripDebugInfoPass());
    }

    // Inline dense resources for now until properly handled throughout the stack
    {
        OpPassManager &funcNestedPM = _pm.nest<func::FuncOp>();
        funcNestedPM.addPass(createDenseResourceInlinerPass());
    }

    if (_options.require_static_shape) {
        OpPassManager &funcNestedPM = _pm.nest<func::FuncOp>();
        funcNestedPM.addPass(createTosaShapedVerificationPass());
    }

    // Type narrowing
    if (_options.type_narrowing != TypeNarrowingMode::None) {
        _pm.addPass(createTypeNarrowingPass({_options.type_narrowing}));
    }

    if (_options.tosa_serialize) {
        tosa_flatbuffer_filename = _options.filename_output;
        if (_options.tosa_fb_schema.empty()) {
            _pm.addPass(mlir::tosa::createTosaSerializePass());
        } else {
            tosa_flatbuffer_schema = _options.tosa_fb_schema;
            OpPassManager &funcNestedPM = _pm.nest<func::FuncOp>();
            funcNestedPM.addPass(mlir::tosa::createTosaSerializeJSONPass());
        }
    } else {
        // Create VGF output
        std::shared_ptr<VGFBuilder> builder = std::make_shared<class VGFBuilder>();

        _pm.nest<func::FuncOp>().addPass(createSignlessIntegerMarkingPass());
        _pm.addPass(createModelPartitionMarkingPass());
        _pm.addPass(createModelPartitioningPass({_options.analysis}));

        {
            OpPassManager &sequenceNestedPM = _pm.nest<vgf::SequenceOp>();
            OpPassManager &segmentNestedPM = sequenceNestedPM.nest<vgf::SegmentOp>();
            OpPassManager &funcNestedPM = segmentNestedPM.nest<func::FuncOp>();

            // NOTE: constant folding should be executed before the convert constants pass
            funcNestedPM.addPass(mlir::tosa::createTosaLayerwiseConstantFoldPass());
            funcNestedPM.addPass(mlir::tosa::createConvertTosaConstantsPass());
        }

        _pm.addPass(createCheckConstantSparsityPass());
        _pm.addPass(createVGFConstantsPass(builder));
        _pm.addPass(mlir::tosa::createTosaToSPIRV(_options.analysis));

        {
            // SPIRV Module Passes
            OpPassManager &sequenceNestedPM = _pm.nest<vgf::SequenceOp>();
            OpPassManager &segmentNestedPM = sequenceNestedPM.nest<vgf::SegmentOp>();
            OpPassManager &spirvNestedPM = segmentNestedPM.nest<spirv::ModuleOp>();
            spirvNestedPM.addPass(mlir::spirv::createSPIRVUpdateVCEPass());
            spirvNestedPM.addPass(mlir::spirv::createSPIRVLowerABIAttributesPass());

            // Sequence Module Passes
            sequenceNestedPM.addPass(
                createSerializeVGFPass(std::move(builder), _options.filename_output, {_options.emit_debug_info}));
        }
    }
}

bool Compiler::Compile(const std::string &input_file) {
    mlir::ParserConfig config(&_context);

    OwningOpRef<ModuleOp> moduleOp;

    if (isTosaFlatbuffer(input_file)) {
        moduleOp = mlir::tosa::BuildMlirFromTosaFile(input_file.c_str(), &_context, true);
    } else {
        moduleOp = OwningOpRef<ModuleOp>(mlir::parseSourceFile<ModuleOp>(input_file, config));
    }

    if (!moduleOp) {
        return false;
    }

    return mlir::succeeded(_pm.run(*moduleOp));
}

} // namespace mlsdk::model_converter
