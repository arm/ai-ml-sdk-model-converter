#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

set(BUILD_TESTS OFF)
set(MLIR_TOSA_OPT OFF)
set(TOSA_ENABLE_PROJECTS "mlir_translator" CACHE STRING "" FORCE)

include_directories(SYSTEM ${TOSA_TOOLS_PATH}/mlir_translator ${CMAKE_BINARY_DIR}/tosa_tools/mlir_translator)

add_subdirectory(${TOSA_TOOLS_PATH}
    ${CMAKE_BINARY_DIR}/tosa_tools SYSTEM EXCLUDE_FROM_ALL)
