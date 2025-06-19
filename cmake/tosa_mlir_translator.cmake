#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

set(BUILD_TESTS OFF)
set(MLIR_TOSA_OPT OFF)

include_directories(SYSTEM ${TOSA_MLIR_TRANSLATOR_PATH} ${CMAKE_BINARY_DIR}/tosa_mlir_translator)

# workaround the strange tablegen path in tosa_mlir_translator which has an "include" prefix
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tosa_mlir_translator/include)
add_subdirectory(${TOSA_MLIR_TRANSLATOR_PATH}
    ${CMAKE_BINARY_DIR}/tosa_mlir_translator SYSTEM EXCLUDE_FROM_ALL)
