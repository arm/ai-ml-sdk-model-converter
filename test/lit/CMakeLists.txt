#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

find_program(MODEL_CONVERTER_LIT_EXECUTABLE
    NAMES
        llvm-lit
        llvm-lit.py
    HINTS
        ${LLVM_TOOLS_BINARY_DIR}
    REQUIRED
)
get_filename_component(MODEL_CONVERTER_LIT_TOOLS_DIR ${MODEL_CONVERTER_LIT_EXECUTABLE} DIRECTORY)
set(LLVM_EXTERNAL_LIT "${MODEL_CONVERTER_LIT_EXECUTABLE}" CACHE STRING "Command used to spawn lit" FORCE)

configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-model-converter-lit "Running ML SDK Model Converter lit tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
        FileCheck
        count
        model-converter
        model-converter-opt
        not
)
set_target_properties(check-model-converter-lit PROPERTIES FOLDER "Tests")
