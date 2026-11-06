#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

include_guard(GLOBAL)

set(CROSS_TOOLCHAIN_FLAGS_LLVM_NATIVE "" CACHE STRING "Toolchain configuration for LLVM_NATIVE")
list(PREPEND CROSS_TOOLCHAIN_FLAGS_LLVM_NATIVE
    "-DCMAKE_PROJECT_LLVM_INCLUDE_BEFORE=${CMAKE_CURRENT_LIST_FILE}")

function(model_converter_remove_llvm_dependency target dependency)
    if(NOT TARGET ${target})
        return()
    endif()

    foreach(property LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
        get_target_property(dependencies ${target} ${property})
        if(dependencies)
            list(REMOVE_ITEM dependencies "${dependency}" "$<LINK_ONLY:${dependency}>")
            set_property(TARGET ${target} PROPERTY ${property} "${dependencies}")
        endif()
    endforeach()
endfunction()

function(model_converter_reduce_llvm_dependencies)
    model_converter_remove_llvm_dependency(MLIRTosaTransforms MLIRFuncTransformOps)
    model_converter_remove_llvm_dependency(MLIRSPIRVSerialization MLIRTranslateLib)
    model_converter_remove_llvm_dependency(mlir-linalg-ods-yaml-gen LLVMCore)
endfunction()

# Run after LLVM has defined its targets.
cmake_language(DEFER CALL model_converter_reduce_llvm_dependencies)
