/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "include/type_narrowing.hpp"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/SourceMgr.h"

#include <string>

using namespace mlir;
using namespace mlir::model_converter_passes;

namespace mlsdk::model_converter {

class Compiler {
  public:
    struct Options {
        std::string filename_output;
        std::string tosa_fb_schema = "";
        bool tosa_serialize = false;
        bool enable_verifier = false;
        bool enable_statistics = false;
        bool dump_mlir = false;
        bool emit_debug_info = false;
        bool require_static_shape = false;
        bool analysis = false;
        // Optimizations
        TypeNarrowingMode type_narrowing = TypeNarrowingMode::None;
    };

    explicit Compiler(const Options &options);

    bool Compile(const std::string &input_file);

    void SetLogging();
    void SetRegistry();
    void SetMultiThreading(bool enable = false);
    void SetPassManager();

  private:
    MLIRContext _context;
    DialectRegistry _registry;
    PassManager _pm;
    llvm::SourceMgr _sourceMgr;
    SourceMgrDiagnosticHandler _sourceMgrHandler;
    Options _options;
};

} // namespace mlsdk::model_converter
