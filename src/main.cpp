/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#include <argparse/argparse.hpp>
#include <vgf/logging.hpp>

#include "llvm/ADT/STLExtras.h"

#include "compiler.hpp"
#include "version.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
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

constexpr std::string_view bespokeCustomOpDomain = "com.arm.bespoke";

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

std::unique_ptr<argparse::ArgumentParser> createParser(int argc, const char *argv[]) {
    std::unique_ptr<argparse::ArgumentParser> parser = nullptr;
    try {
        parser = std::make_unique<argparse::ArgumentParser>(argv[0], details::version);

        parser->add_argument("-i", "--input")
            .help("the input file to read TOSA FlatBuffer or TOSA MLIR data from")
            .default_value(std::string{"-"});
        parser->add_argument("-o", "--output")
            .help("the output file to write VGF data to")
            .default_value(std::string{"-"});
        parser->add_argument("--tosa-flatbuffer")
            .help("write tosa FlatBuffer instead of VGF")
            .default_value(false)
            .implicit_value(true);
        parser->add_argument("--tosa-flatbuffer-schema")
            .help("path to the tosa FlatBuffer schema")
            .default_value(std::string{""});
        parser->add_argument("--dump-mlir")
            .help("Dump MLIR between each pass to std error")
            .default_value(false)
            .implicit_value(true);
        parser->add_argument("--emit-debug-info")
            .help("Produce debug info instructions in SPIR-V assembly")
            .default_value(false)
            .implicit_value(true);
        parser->add_argument("--disable-replicated-composites")
            .help("Do not emit SPIR-V features that require VK_EXT_shader_replicated_composites")
            .default_value(false)
            .implicit_value(true);
        parser->add_argument("--require-static-shape")
            .help("Require all tensors to be ranked and have a specified shape. Terminate on deviation.")
            .default_value(false)
            .implicit_value(true);
        parser->add_argument("--experimental-analysis")
            .help("Print analysis output (what operator lower and which errors out) for the input. [EXPERIMENTAL]")
            .default_value(false)
            .implicit_value(true);
        parser->add_argument("--custom-op-domain-to-opcode")
            .help("Map TOSA custom op domains to Arm.ExperimentalMLOperations CALL Opcode literal integers. Entries "
                  "use <domain>:<opcode> and can be specified multiple times.")
            .append()
            .default_value(std::vector<std::string>{});
        parser->add_argument("--enable-bespoke")
            .help("Enable com.arm.bespoke custom op lowering as an Arm.ExperimentalMLOperations CALL with Opcode 0.")
            .default_value(false)
            .implicit_value(true);
        // Optimisation Options
        parser->add_argument("--type-narrowing")
            .help("Perform type-narrowing to all operator operands/results from fp32 -> fp16")
            .default_value("none")
            .choices("none", "full", "partial", "full_preserve_io");
        parser->parse_args(argc, argv);
        return parser;
    } catch (const std::exception &err) {
        llvm::errs() << err.what() << "\n";
        return nullptr;
    }
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
    std::unique_ptr<argparse::ArgumentParser> parser = createParser(argc, argv);
    if (parser == nullptr) {
        llvm::errs() << "Error creating parser\n";
        return -1;
    }

    std::string input;
    Compiler::Options options;
    options.enable_verifier = true;
    options.enable_statistics = false;

    try {
        input = parser->get("--input");
        options.filename_output = parser->get("--output");
        options.tosa_serialize = parser->get<bool>("--tosa-flatbuffer");
        options.tosa_fb_schema = options.tosa_serialize ? parser->get("--tosa-flatbuffer-schema") : "";
        options.dump_mlir = parser->get<bool>("--dump-mlir");
        options.emit_debug_info = parser->get<bool>("--emit-debug-info");
        options.disable_replicated_composites = parser->get<bool>("--disable-replicated-composites");
        options.require_static_shape = parser->get<bool>("--require-static-shape");
        options.analysis = parser->get<bool>("--experimental-analysis");
        options.custom_op_domain_to_opcode = parser->get<std::vector<std::string>>("--custom-op-domain-to-opcode");
        if (!validateCustomOpDomainToOpcode(options.custom_op_domain_to_opcode)) {
            return -1;
        }
        if (parser->get<bool>("--enable-bespoke")) {
            options.custom_op_domain_to_opcode.push_back(std::string(bespokeCustomOpDomain) + ":0");
        }
        const std::string typeNarrowing = parser->get("--type-narrowing");
        if (typeNarrowing == "full") {
            options.type_narrowing = TypeNarrowingMode::Full;
        } else if (typeNarrowing == "partial") {
            options.type_narrowing = TypeNarrowingMode::Partial;
        } else if (typeNarrowing == "full_preserve_io") {
            options.type_narrowing = TypeNarrowingMode::FullPreserveIO;
        } else {
            options.type_narrowing = TypeNarrowingMode::None;
        }
    } catch (const std::exception &err) {
        llvm::errs() << err.what() << "\n";
        try {
            std::ostringstream oss;
            oss << *parser;
            llvm::errs() << oss.str() << "\n";
        } catch (...) {
            llvm::errs() << "Unable to print parser\n";
        }
        return -1;
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
