/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#include <vgf/logging.hpp>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"

#include "compiler.hpp"
#include "version.hpp"

#include <fstream>
#include <string_view>
#include <vector>

#ifdef _WIN32
#    include <io.h>
#else
#    include <unistd.h>
#endif

using namespace mlsdk::model_converter;
using namespace mlsdk::vgflib;

namespace {

namespace cl = llvm::cl;

constexpr std::string_view bespokeCustomOpDomain = "com.arm.bespoke";

struct CommandLineOptions {
    cl::OptionCategory category{"Model Converter options"};
    cl::opt<std::string> input{"input", cl::desc("the input file to read TOSA FlatBuffer or TOSA MLIR data from"),
                               cl::init("-"), cl::cat(category)};
    cl::alias inputAlias{"i", cl::desc("Alias for --input"), cl::aliasopt(input), cl::cat(category), cl::NotHidden};
    cl::opt<std::string> output{"output", cl::desc("the output file to write VGF data to"), cl::init("-"),
                                cl::cat(category)};
    cl::alias outputAlias{"o", cl::desc("Alias for --output"), cl::aliasopt(output), cl::cat(category), cl::NotHidden};
    cl::opt<bool> tosaFlatbuffer{"tosa-flatbuffer", cl::desc("write tosa FlatBuffer instead of VGF"),
                                 cl::ValueDisallowed, cl::cat(category)};
    cl::opt<std::string> tosaFlatbufferSchema{"tosa-flatbuffer-schema", cl::desc("path to the tosa FlatBuffer schema"),
                                              cl::cat(category)};
#ifdef BLOCK_SCALED_TENSOR
    cl::opt<std::string> targetTosaVersion{
        "target-tosa-version",
        cl::desc("Set the target TOSA import ceiling for VGF output (ignored with --tosa-flatbuffer)"),
        cl::cat(category)};
#endif
    cl::opt<bool> dumpMlir{"dump-mlir", cl::desc("Dump MLIR between each pass to std error"), cl::ValueDisallowed,
                           cl::cat(category)};
    cl::opt<bool> emitDebugInfo{"emit-debug-info", cl::desc("Produce debug info instructions in SPIR-V assembly"),
                                cl::ValueDisallowed, cl::cat(category)};
    cl::opt<bool> disableReplicatedComposites{
        "disable-replicated-composites",
        cl::desc("Do not emit SPIR-V features that require VK_EXT_shader_replicated_composites"), cl::ValueDisallowed,
        cl::cat(category)};
    cl::opt<bool> requireStaticShape{
        "require-static-shape",
        cl::desc("Require all tensors to be ranked and have a specified shape. Terminate on deviation."),
        cl::ValueDisallowed, cl::cat(category)};
    cl::opt<bool> experimentalAnalysis{
        "experimental-analysis",
        cl::desc("Print analysis output (what operator lower and which errors out) for the input. [EXPERIMENTAL]"),
        cl::ValueDisallowed, cl::cat(category)};
    cl::list<std::string> customOpDomainToOpcode{
        "custom-op-domain-to-opcode",
        cl::desc("Map TOSA custom op domains to Arm.ExperimentalMLOperations CALL Opcode literal integers. Entries "
                 "use <domain>:<opcode> and can be specified multiple times."),
        cl::ZeroOrMore, cl::cat(category)};
    cl::opt<bool> enableBespoke{
        "enable-bespoke",
        cl::desc("Enable com.arm.bespoke custom op lowering as an Arm.ExperimentalMLOperations CALL with Opcode 0."),
        cl::ValueDisallowed, cl::cat(category)};
    cl::opt<TypeNarrowingMode> typeNarrowing{
        "type-narrowing", cl::desc("Perform type-narrowing to all operator operands/results from fp32 -> fp16"),
        cl::values(clEnumValN(TypeNarrowingMode::None, "none", "Disable type narrowing"),
                   clEnumValN(TypeNarrowingMode::Full, "full", "Narrow all supported operations"),
                   clEnumValN(TypeNarrowingMode::Partial, "partial", "Narrow supported operations selectively"),
                   clEnumValN(TypeNarrowingMode::FullPreserveIO, "full_preserve_io", "Preserve input/output types")),
        cl::init(TypeNarrowingMode::None), cl::cat(category)};
};

bool validateCustomOpDomainToOpcode(const std::vector<std::string> &mappings) {
    auto reservedMapping = llvm::find_if(mappings, [](const std::string &mapping) {
        const std::size_t delimiter = mapping.rfind(':');
        return delimiter != std::string::npos &&
               std::string_view(mapping).substr(0, delimiter) == bespokeCustomOpDomain;
    });

    if (reservedMapping == mappings.end()) {
        return true;
    }

    llvm::errs() << "Bad custom op domain mapping: " << bespokeCustomOpDomain << " is reserved. "
                 << "Use --enable-bespoke instead of --custom-op-domain-to-opcode " << *reservedMapping << "\n";
    return false;
}

void vgfLoggingHandler(logging::LogLevel vgfLogLevel, const std::string &message) {
    std::string logLevel;
    switch (vgfLogLevel) {
    case logging::LogLevel::INFO:
        logLevel = "info";
        break;
    case logging::LogLevel::DEBUG:
        logLevel = "debug";
        break;
    case logging::LogLevel::WARNING:
        logLevel = "warning";
        break;
    case logging::LogLevel::ERROR:
        logLevel = "error";
        break;
    default:
        logLevel = "unknown";
    }
    llvm::errs() << "VGF [" << logLevel << "]: " << message << "\n";
}

} // namespace

int main(int argc, const char *argv[]) {
    // Dependencies register global options, including tosa-flatbuffer-schema.
    // MC owns its command line and configures compiler passes explicitly.
    cl::ResetCommandLineParser();
    // Construction registers the options and their defaults with LLVM.
    // ParseCommandLineOptions below populates args from argv.
    CommandLineOptions args;
    cl::SetVersionPrinter([](llvm::raw_ostream &stream) { stream << details::version << "\n"; });
    cl::alias versionAlias{"v", cl::desc("Alias for --version"),
                           cl::aliasopt(*cl::getRegisteredOptions().lookup("version")), cl::cat(args.category),
                           cl::NotHidden};
    cl::HideUnrelatedOptions(args.category);
    if (!cl::ParseCommandLineOptions(argc, argv, "ML SDK Model Converter\n", &llvm::errs())) {
        return -1;
    }

    const std::string &input = args.input.getValue();
    Compiler::Options options;
    options.enable_verifier = true;
    options.filename_output = args.output;
    options.tosa_serialize = args.tosaFlatbuffer;
    options.tosa_fb_schema = options.tosa_serialize ? args.tosaFlatbufferSchema.getValue() : "";
#ifdef BLOCK_SCALED_TENSOR
    if (args.targetTosaVersion.getNumOccurrences() > 0) {
        options.target_tosa_version = args.targetTosaVersion.getValue();
    }
#endif
    options.dump_mlir = args.dumpMlir;
    options.emit_debug_info = args.emitDebugInfo;
    options.disable_replicated_composites = args.disableReplicatedComposites;
    options.require_static_shape = args.requireStaticShape;
    options.analysis = args.experimentalAnalysis;
    options.custom_op_domain_to_opcode.assign(args.customOpDomainToOpcode.begin(), args.customOpDomainToOpcode.end());
    options.type_narrowing = args.typeNarrowing;
    if (!validateCustomOpDomainToOpcode(options.custom_op_domain_to_opcode)) {
        return -1;
    }
    if (args.enableBespoke) {
        options.custom_op_domain_to_opcode.push_back(std::string(bespokeCustomOpDomain) + ":0");
    }

    if (input == "-") {
#ifdef _WIN32
        if (_isatty(0)) {
#else
        if (isatty(0)) {
#endif
            llvm::errs() << "Warning: Using terminal input\n";
        }
    }

    if (options.filename_output != "-" && input == options.filename_output) {
        llvm::errs() << "Bad output filename: Won't overwrite input\n";
        return -1;
    }

    logging::EnableLogging(vgfLoggingHandler);

    std::ifstream outputFile(options.filename_output);
    if (outputFile.good()) {
        llvm::errs() << "Warning: will overwrite existing output file: " << options.filename_output << "\n";
    }

    Compiler compiler(options);
    compiler.SetLogging();
    compiler.SetRegistry();
    compiler.SetMultiThreading(false);
    compiler.SetPassManager();

    if (!compiler.Compile(input)) {
        llvm::errs() << "Failed to compile network\n";
        return -1;
    }

    return 0;
}
