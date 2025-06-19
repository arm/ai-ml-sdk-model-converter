/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#include <argparse/argparse.hpp>
#include <vgf/logging.hpp>

#include "compiler.hpp"
#include "version.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

using namespace mlsdk::model_converter;
using namespace mlsdk::vgflib;

namespace {

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
        parser->add_argument("--require-static-shape")
            .help("Require all tensors to be ranked and have a specified shape. Terminate on deviation.")
            .default_value(false)
            .implicit_value(true);
        parser->add_argument("--experimental-analysis")
            .help("Print analysis output (what operator lower and which errors out) for the input. [EXPERIMENTAL]")
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
        options.require_static_shape = parser->get<bool>("--require-static-shape");
        options.analysis = parser->get<bool>("--experimental-analysis");
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
