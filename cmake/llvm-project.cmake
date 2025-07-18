#
# SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

set(LLVM_PATH "LLVM-NOTFOUND" CACHE PATH "Path to LLVM")
set(LLVM_VERSION "unknown")
set(MLIR_VERSION "unknown")

# Recursively collect compiled library targets
function(mlsdk_get_compiled_targets dir collected_targets)
    get_directory_property(DIR_TARGETS DIRECTORY ${dir} BUILDSYSTEM_TARGETS)

    foreach(target ${DIR_TARGETS})
        get_target_property(target_type ${target} TYPE)
        if(target_type STREQUAL STATIC_LIBRARY OR target_type STREQUAL SHARED_LIBRARY OR
            target_type STREQUAL OBJECT_LIBRARY)
            list(APPEND ${collected_targets} ${target})
        endif()
    endforeach()

    # Recursively process subdirectories
    get_directory_property(SUBDIRS DIRECTORY ${dir} SUBDIRECTORIES)
    foreach(subdir ${SUBDIRS})
        mlsdk_get_compiled_targets(${subdir} ${collected_targets})
    endforeach()

    # Pass updated list to the parent scope
    set(${collected_targets} "${${collected_targets}}" PARENT_SCOPE)
endfunction()

if(EXISTS ${LLVM_PATH}/llvm/CMakeLists.txt)

    # Try applying LLVM changes patch
    if(LLVM_PROJECT_PATCH_FILE)
        execute_process(
            COMMAND git apply --check "${LLVM_PROJECT_PATCH_FILE}"
            WORKING_DIRECTORY "${LLVM_PATH}"
            RESULT_VARIABLE LLVM_PATCH_APPLY_RESULT
            ERROR_QUIET)
        if(LLVM_PATCH_APPLY_RESULT EQUAL 0)
            message(STATUS "Applying LLVM changes patch")
            execute_process(
                COMMAND git apply "${LLVM_PROJECT_PATCH_FILE}"
                WORKING_DIRECTORY "${LLVM_PATH}")
        else()
            message(STATUS "LLVM changes patch is not applicable or failed to apply")
        endif()
    endif()

    if(NOT TARGET LLVMCore)
        message(STATUS "Building LLVM and MLIR from source")

        set(TARGETS_TO_BUILD "all")
        # We aim to build only for the specific target architectures of interest to us.
        # This is due to building all target architectures adds an LLVM target named "SPIRV",
        # which clashes with the glslang SPIRV target
        if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "^(x86|x86_64|AMD64|x64)$")
            set(TARGETS_TO_BUILD "X86")
        elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "^arm64")
            set(TARGETS_TO_BUILD "AArch64")
        endif()

        set(LLVM_ABI_BREAKING_CHECKS "FORCE_OFF" CACHE STRING "")
        set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
        set(LLVM_ENABLE_EH ON CACHE BOOL "")
        set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "")
        set(LLVM_ENABLE_RTTI ON CACHE BOOL "")
        set(LLVM_ENABLE_TERMINFO OFF CACHE BOOL "")
        set(LLVM_PARALLEL_LINK_JOBS "2" CACHE STRING "")
        set(LLVM_REQUIRES_EH ON CACHE BOOL "")
        set(LLVM_REQUIRES_RTTI ON CACHE BOOL "")
        set(LLVM_TARGETS_TO_BUILD "${TARGETS_TO_BUILD}" CACHE STRING "")

        add_subdirectory(${LLVM_PATH}/llvm llvm SYSTEM EXCLUDE_FROM_ALL)

        set(ALL_COMPILED_TARGETS)
        mlsdk_get_compiled_targets(${LLVM_PATH}/llvm ALL_COMPILED_TARGETS)
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

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
