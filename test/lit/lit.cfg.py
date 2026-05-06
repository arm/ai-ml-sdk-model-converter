# -*- Python -*-
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import lit.formats
from lit.llvm import llvm_config

config.name = "MODEL-CONVERTER"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]

config.test_source_root = config.model_converter_lit_source_dir
config.test_exec_root = config.model_converter_lit_binary_dir

config.excludes = [
    "Inputs",
    "CMakeLists.txt",
    "lit.cfg.py",
    "lit.site.cfg.py",
]

llvm_config.with_system_environment(["HOME", "TMP", "TEMP"])
llvm_config.use_default_substitutions()

tool_dirs = [
    config.model_converter_tools_dir,
    config.llvm_tools_dir,
]

tools = [
    "model-converter",
    "model-converter-opt",
    "FileCheck",
    "not",
    "count",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
