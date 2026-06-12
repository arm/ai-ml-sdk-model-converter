#
# SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

set(TOSA_TOOLS_PATH "TOSA_TOOLS-NOTFOUND" CACHE PATH "Path to TOSA Tools")
set(tosa_serialize_VERSION "unknown")

if(EXISTS ${TOSA_TOOLS_PATH}/CMakeLists.txt)
    mlsdk_get_git_revision(${TOSA_TOOLS_PATH} tosa_serialize_VERSION)
    if(NOT TARGET tosa_serialize)
        set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(MLIR_TOSA_OPT OFF CACHE BOOL "" FORCE)
        set(TOSA_BUILD_GRAPH_TOOLS OFF CACHE BOOL "" FORCE)
        set(TOSA_ENABLE_PROJECTS "mlir_translator" CACHE STRING "" FORCE)

        include_directories(SYSTEM ${TOSA_TOOLS_PATH}/mlir_translator ${CMAKE_BINARY_DIR}/tosa_tools/mlir_translator)

        add_subdirectory(${TOSA_TOOLS_PATH}
            ${CMAKE_BINARY_DIR}/tosa_tools SYSTEM EXCLUDE_FROM_ALL)
    endif()
else()
    find_package(tosa_serialize REQUIRED CONFIG)
endif()
