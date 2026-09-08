/*
 * SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "compiler.hpp"
#include "include/passes.hpp"
#include "mlir/Conversion/TosaToSPIRVTosa/TosaToSPIRVTosa.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "vgf-dialect/VGFDialect.h"
#include "vgf_builder.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include "include/DeserializationPasses.h" // from @tosa_tools/mlir_translator
#include "include/SerializationPasses.h"   // from @tosa_tools/mlir_translator

#include <array>
#include <fstream>
#include <iostream>
#include <string_view>

using namespace mlir::model_converter_passes;

namespace mlsdk::model_converter {

namespace {
bool isTosaFlatbuffer(const std::string &input) {
    std::ifstream stream(input, std::ios::binary);

    // Non-size-prefixed FlatBuffers store their four-byte file identifier after the
    // four-byte root table offset. The TOSA schema declares this identifier as "TOSA".
    std::array<char, 8> header{};
    // Reading the complete header also rejects files shorter than eight bytes.
    if (!stream.read(header.data(), header.size())) {
        return false;
    }

    return std::string_view(header.data() + 4, 4) == "TOSA";
}

LogicalResult printPassOptionError(const Twine &message) {
    llvm::errs() << message << "\n";
    return failure();
}

std::unique_ptr<Pass> createConfiguredTosaSerializeJSONPass(const std::string &filename, const std::string &schema) {
    auto pass = mlir::tosa::createTosaSerializeJSONPass();
    std::string passOptions = "tosa-flatbuffer-filename=" + filename + " tosa-flatbuffer-schema=" + schema;
    if (failed(pass->initializeOptions(passOptions, printPassOptionError))) {
        llvm::report_fatal_error("Failed to configure TOSA JSON serialization pass");
    }
    return pass;
}
std::unique_ptr<Pass> createConfiguredTosaToSPIRVTosaPass(bool analysis,
                                                          const std::vector<std::string> &customOpDomainToOpcode) {
    auto pass = mlir::tosa::createTosaToSPIRVTosa(analysis);
    if (customOpDomainToOpcode.empty()) {
        return pass;
    }

    std::string passOptions = "custom-op-domain-to-opcode=";
    llvm::raw_string_ostream optionsStream(passOptions);
    llvm::interleave(customOpDomainToOpcode, optionsStream, ",");
    optionsStream.flush();

    if (failed(pass->initializeOptions(passOptions, printPassOptionError))) {
        llvm::report_fatal_error("Failed to configure TOSA to SPIR-V TOSA pass");
    }
    return pass;
}
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

    {
        OpPassManager &funcNestedPM = _pm.nest<func::FuncOp>();
        funcNestedPM.addPass(mlir::tosa::createTosaConvertIntegerTypeToSignless());
        // Inline dense resources for now until properly handled throughout the stack
        funcNestedPM.addPass(createDenseResourceInlinerPass());
    }

    if (_options.require_static_shape) {
        OpPassManager &funcNestedPM = _pm.nest<func::FuncOp>();
        funcNestedPM.addPass(createTosaShapedVerificationPass());
    }

    // Canonicalize only after validating the input so that dead, malformed IR cannot be
    // removed before the verifier sees it. Doing this before narrowing also avoids
    // rewriting operations that can already be folded or eliminated. Keep this pass on
    // func.func: canonicalization also removes trivially dead, side-effect-free operations
    // without pruning the module's input/output interfaces. Do not replace it with
    // createRemoveDeadValuesPass(), which may remove function arguments or results. If
    // unreachable private functions ever need removing, add createSymbolDCEPass() at the
    // module level here, after validation and before narrowing.
    _pm.nest<func::FuncOp>().addPass(mlir::createCanonicalizerPass());

    // Type narrowing
    if (_options.type_narrowing != TypeNarrowingMode::None) {
        _pm.addPass(createTypeNarrowingPass({_options.type_narrowing}));
    }

    {
        OpPassManager &funcNestedPM = _pm.nest<func::FuncOp>();
        funcNestedPM.addPass(mlir::tosa::createTosaNarrowI64ToI32Pass({true, true}));
        funcNestedPM.addPass(mlir::tosa::createTosaNarrowF64ToF32Pass({true, true}));

        // Dialect conversion and type narrowing can expose new folds or leave redundant
        // casts and dead operations. Clean those up before signless interface metadata,
        // graph constant IDs, and partition plans are computed below. Moving this pass
        // past any of those stages can make their annotations and bookkeeping stale.
        funcNestedPM.addPass(mlir::createCanonicalizerPass());
    }

    if (_options.tosa_serialize) {
        if (_options.tosa_fb_schema.empty()) {
            _pm.addPass(mlir::tosa::createTosaSerializePass(_options.filename_output));
        } else {
            _pm.addPass(createConfiguredTosaSerializeJSONPass(_options.filename_output, _options.tosa_fb_schema));
        }
    } else {
        // Create VGF output
        std::shared_ptr<VGFBuilder> builder = std::make_shared<class VGFBuilder>();

        _pm.nest<func::FuncOp>().addPass(createSignlessIntegerMarkingPass());
        {
            OpPassManager &funcNestedPM = _pm.nest<func::FuncOp>();
            // This pass registers the canonicalization patterns for every TOSA operation
            // and applies them greedily in addition to its specialized constant folds. It
            // therefore serves as the VGF pipeline's later, TOSA-specific canonicalization
            // phase; do not add a redundant generic canonicalizer immediately around it.
            // Run constant folding before assigning graph constant IDs so fold-created constants get stable,
            // sequence-wide IDs before partitioning clones them into graph segments.
            funcNestedPM.addPass(mlir::tosa::createTosaLayerwiseConstantFoldPass());
            funcNestedPM.addPass(mlir::tosa::createTosaToSPIRVTosaMarkGraphConstants());
        }
        _pm.addPass(createModelPartitionMarkingPass());
        _pm.addPass(createModelPartitioningPass({_options.analysis}));

        _pm.addPass(createCheckConstantSparsityPass());
        _pm.addPass(createVGFConstantsPass(builder));
        _pm.nest<vgf::SequenceOp>().addPass(createAssignGraphARMInterfaceVarABIPass());
        _pm.addPass(createConfiguredTosaToSPIRVTosaPass(_options.analysis, _options.custom_op_domain_to_opcode));

        {
            // SPIRV Module Passes
            OpPassManager &sequenceNestedPM = _pm.nest<vgf::SequenceOp>();
            OpPassManager &segmentNestedPM = sequenceNestedPM.nest<vgf::SegmentOp>();
            OpPassManager &spirvNestedPM = segmentNestedPM.nest<spirv::ModuleOp>();
            if (!_options.disable_replicated_composites) {
                spirvNestedPM.addPass(mlir::spirv::createSPIRVReplicatedConstantCompositePass());
            }
            spirvNestedPM.addPass(mlir::spirv::createSPIRVLowerABIAttributesPass());
            spirvNestedPM.addPass(mlir::spirv::createSPIRVUpdateVCEPass());

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
