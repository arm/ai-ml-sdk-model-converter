#
# SPDX-FileCopyrightText: Copyright 2022-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

set(LLVM_PATH "LLVM-NOTFOUND" CACHE PATH "Path to LLVM")
set(LLVM_VERSION "unknown")
set(MLIR_VERSION "unknown")

option(MODEL_CONVERTER_APPLY_LLVM_PATCH "Apply LLVM patch" ON)
# The internal LLVM used without the bundled patch provides block-scaled tensors.
if(MODEL_CONVERTER_APPLY_LLVM_PATCH)
    set(BLOCK_SCALED_TENSOR OFF)
else()
    set(BLOCK_SCALED_TENSOR ON)
endif()
set(BLOCK_SCALED_TENSOR ${BLOCK_SCALED_TENSOR} CACHE INTERNAL "Block-scaled tensor support" FORCE)

if(EXISTS ${LLVM_PATH}/llvm/CMakeLists.txt)
    if(MODEL_CONVERTER_APPLY_LLVM_PATCH)
        set(LLVM_PROJECT_PATCH_FILES
            "${CMAKE_CURRENT_LIST_DIR}/../patches/llvm.patch"
        )

        if(LLVM_PROJECT_PATCH_FILES)
            find_package(Git REQUIRED)

            foreach(LLVM_PROJECT_PATCH_FILE IN LISTS LLVM_PROJECT_PATCH_FILES)
                execute_process(
                    COMMAND
                        "${GIT_EXECUTABLE}"
                        -C
                        "${LLVM_PATH}"
                        apply
                        --reverse
                        --check
                        "${LLVM_PROJECT_PATCH_FILE}"
                    RESULT_VARIABLE LLVM_PATCH_REVERSE_CHECK
                    OUTPUT_VARIABLE LLVM_PATCH_REVERSE_CHECK_OUTPUT
                    ERROR_VARIABLE LLVM_PATCH_REVERSE_CHECK_ERROR
                )

                if(LLVM_PATCH_REVERSE_CHECK EQUAL 0)
                    message(STATUS "LLVM patch ${LLVM_PROJECT_PATCH_FILE} is already applied")
                    continue()
                endif()

                execute_process(
                    COMMAND
                        "${GIT_EXECUTABLE}"
                        -C
                        "${LLVM_PATH}"
                        -c
                        user.name=svc_sdk
                        -c
                        user.email=svc_sdk@arm.com
                        am
                        "${LLVM_PROJECT_PATCH_FILE}"
                    RESULT_VARIABLE LLVM_APPLY_AND_COMMIT_PATCH
                    OUTPUT_VARIABLE LLVM_APPLY_AND_COMMIT_PATCH_OUTPUT
                    ERROR_VARIABLE LLVM_APPLY_AND_COMMIT_PATCH_ERROR
                )
                if(LLVM_APPLY_AND_COMMIT_PATCH EQUAL 0)
                    execute_process(
                        COMMAND "${GIT_EXECUTABLE}" -C "${LLVM_PATH}" log -1 --oneline
                        OUTPUT_VARIABLE LLVM_PATCH_COMMIT
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                    )
                    message(STATUS "LLVM patch ${LLVM_PATCH_COMMIT} applied")
                else()
                    execute_process(
                        COMMAND "${GIT_EXECUTABLE}" -C "${LLVM_PATH}" am --abort
                        OUTPUT_QUIET
                        ERROR_QUIET
                    )
                    message(STATUS "${LLVM_PATCH_REVERSE_CHECK}")
                    message(STATUS "${LLVM_PATCH_REVERSE_CHECK_OUTPUT}")
                    message(STATUS "${LLVM_PATCH_REVERSE_CHECK_ERROR}")
                    message(STATUS "${LLVM_APPLY_AND_COMMIT_PATCH}")
                    message(STATUS "${LLVM_APPLY_AND_COMMIT_PATCH_OUTPUT}")
                    message(STATUS "${LLVM_APPLY_AND_COMMIT_PATCH_ERROR}")
                    message(FATAL_ERROR "Failed to apply LLVM patch ${LLVM_PROJECT_PATCH_FILE}")
                endif()
            endforeach()
        endif()
    endif()

    if(NOT TARGET LLVMCore)
        message(STATUS "Building LLVM and MLIR from source")

        # SPIR-V is emitted by MLIR's serializer, without an LLVM code generator
        # or JIT. Conversion patterns are implemented in C++ and do not use PDL.
        set(LLVM_TARGETS_TO_BUILD "" CACHE STRING "LLVM target architectures to build")
        set(MLIR_ENABLE_EXECUTION_ENGINE OFF CACHE BOOL "Enable the MLIR execution engine")
        set(MLIR_ENABLE_PDL_IN_PATTERNMATCH OFF CACHE BOOL "Enable PDL in MLIR pattern matching")

        set(LLVM_ABI_BREAKING_CHECKS "FORCE_OFF" CACHE STRING "")
        set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
        set(LLVM_ENABLE_EH ON CACHE BOOL "")
        set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "")
        set(LLVM_ENABLE_RTTI ON CACHE BOOL "")
        set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
        # Avoid an optional host zstd dependency in distributed binaries.
        set(LLVM_ENABLE_ZSTD OFF CACHE STRING "")
        set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
        set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
        if(MODEL_CONVERTER_BUILD_TESTS)
            set(LLVM_INCLUDE_UTILS ON CACHE BOOL "" FORCE)
        else()
            set(LLVM_INCLUDE_UTILS OFF CACHE BOOL "" FORCE)
        endif()
        set(LLVM_BUILD_UTILS OFF CACHE BOOL "" FORCE)
        set(LLVM_TOOL_LLVM_C_TEST_BUILD OFF CACHE BOOL "" FORCE)
        set(MLIR_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
        set(LLVM_PARALLEL_LINK_JOBS "2" CACHE STRING "")
        set(LLVM_REQUIRES_EH ON CACHE BOOL "")
        set(LLVM_REQUIRES_RTTI ON CACHE BOOL "")

        include(${CMAKE_CURRENT_LIST_DIR}/llvm-deps-cleanup.cmake)
        add_subdirectory(${LLVM_PATH}/llvm llvm SYSTEM EXCLUDE_FROM_ALL)
    endif()
endif()

find_package(MLIR REQUIRED CONFIG HINTS "${CMAKE_BINARY_DIR}/lib/cmake" "${LLVM_PATH}")
find_package(LLVM REQUIRED CONFIG HINTS "${CMAKE_BINARY_DIR}/lib/cmake" "${LLVM_PATH}")

list(APPEND CMAKE_MODULE_PATH
    ${LLVM_CMAKE_DIR}
    ${MLIR_CMAKE_DIR})

include(AddLLVM)
include(TableGen)
include(AddMLIR)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_MAIN_SRC_DIR})

set(MLIR_TABLEGEN_EXE "mlir-tblgen" CACHE INTERNAL "")
