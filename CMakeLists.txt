#
# SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

cmake_minimum_required(VERSION 3.25)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_LIST_DIR}/cmake/
)

project(ModelConverter
    DESCRIPTION "ML SDK Model Converter"
    LANGUAGES CXX C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Compile defines
add_compile_definitions(
    "$<$<NOT:$<CONFIG:DEBUG>>:NDEBUG>")

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "${CMAKE_PROJECT_NAME} requires an out of source build.")
endif()


#############################################################################
# Options
#############################################################################

option(MODEL_CONVERTER_ENABLE_CCACHE "Enable CCACHE support" OFF)
option(MODEL_CONVERTER_BUILD_DOCS "Build documentation" OFF)
option(MODEL_CONVERTER_ENABLE_COVERAGE "Enable coverage build" OFF)
option(MODEL_CONVERTER_ENABLE_LTO "Enable Link Time Optimization when CMAKE_BUILD_TYPE=Release" ON)
set(MODEL_CONVERTER_LOG_TRACE_MASK "" CACHE STRING "Log trace enable mask")

set(MODEL_CONVERTER_PACKAGE_NAME ${PROJECT_NAME})
set(MODEL_CONVERTER_NAMESPACE ${PROJECT_NAME})
if(NOT (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR) AND ML_SDK_GENERATE_CPACK)
    set(MODEL_CONVERTER_PACKAGE_NAME "ai_ml_sdk_for_vulkan")
else()
    set(CPACK_PACKAGE_NAME "ai_ml_sdk_model_converter")
endif()

#########################################################################################
# Third Party Dependency path resolver
#########################################################################################

set(TOSA_MLIR_TRANSLATOR_PATH "TOSA_MLIR_TRANSLATOR-NOTFOUND" CACHE PATH "Path to TOSA MLIR Translator")

#############################################################################
# Toolchain configuration
#############################################################################

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM AND MODEL_CONVERTER_ENABLE_CCACHE)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

#############################################################################
# Definitions
#############################################################################

include(argparse)
include(flatbuffers)
include(vgf-lib)
include(llvm-project)
include(version)
include(tosa_mlir_translator)

set(DEFAULT_CMAKE_BUILD_TYPE "Debug")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, default to ${DEFAULT_CMAKE_BUILD_TYPE}")
    set(CMAKE_BUILD_TYPE "${DEFAULT_CMAKE_BUILD_TYPE}" CACHE STRING "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)
endif()

# Make this global - we're building DLLs
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# LTO/IPO support. Only enable in Release mode
if(NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug" AND NOT MODEL_CONVERTER_ENABLE_COVERAGE AND MODEL_CONVERTER_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT MODEL_CONVERTER_LTO_SUPPORTED)
    if(MODEL_CONVERTER_LTO_SUPPORTED)
        message(STATUS "Looking for IPO support - Success")
    else()
        message(STATUS "Looking for IPO support - Not found")
    endif()
else()
    set(MODEL_CONVERTER_LTO_SUPPORTED OFF)
endif()

#############################################################################
# Top level targets, Table Gen Dialect Libraries
#############################################################################

set(LLVM_TARGET_DEFINITIONS src/include/passes.td)
mlir_tablegen(passes.hpp.inc -gen-pass-decls)
add_public_tablegen_target(MLIRModelConverterPasses)

set(LLVM_TARGET_DEFINITIONS src/vgf-dialect/VGFDialect.td)
mlir_tablegen(VGFDialect.h.inc -gen-dialect-decls --dialect=vgf)
mlir_tablegen(VGFDialect.cpp.inc -gen-dialect-defs --dialect=vgf)
add_public_tablegen_target(MLIRVGFDialectIncGen)

set(LLVM_TARGET_DEFINITIONS src/vgf-dialect/VGFOps.td)
mlir_tablegen(VGFOps.h.inc -gen-op-decls)
mlir_tablegen(VGFOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRVGFOpsIncGen)

set(LLVM_TARGET_DEFINITIONS src/vgf-dialect/VGFEnums.td)
mlir_tablegen(VGFAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=vgf)
mlir_tablegen(VGFAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=vgf)
mlir_tablegen(VGFEnums.h.inc -gen-enum-decls)
mlir_tablegen(VGFEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRVGFAttributesIncGen)

set(CONVERTER_LIB_SOURCES
    "src/compiler.cpp"
    "src/conversion/check_sparsity.cpp"
    "src/conversion/dense_resource_inliner.cpp"
    "src/conversion/model_partition_marking.cpp"
    "src/conversion/model_partitioning.cpp"
    "src/conversion/serialize_vgf.cpp"
    "src/conversion/signless_integer_marking.cpp"
    "src/conversion/type_narrowing.cpp"
    "src/conversion/verify_shaped_tensors.cpp"
    "src/conversion/vgf_constants.cpp"
    "src/utils.cpp"
    "src/vgf-dialect/VGFDialect.cpp"
)

set(CONVERTER_LIB_DEPENDENCIES
    flatbuffers::flatbuffers
    MLIRFuncDialect
    MLIRGPUDialect
    MLIRTosaDialect
    MLIRTosaToSPIRV
    MLIRSPIRVDialect
    MLIRSPIRVSerialization
    MLIRSPIRVTransforms
    tosa_serialize
    VGF::vgf
)

add_library(model-converter-lib STATIC ${CONVERTER_LIB_SOURCES})
target_compile_definitions(model-converter-lib PRIVATE
    $<$<BOOL:${MODEL_CONVERTER_LOG_TRACE_MASK}>:LOG_TRACE_ENABLE=${MODEL_CONVERTER_LOG_TRACE_MASK}>
)
target_include_directories(model-converter-lib PUBLIC src)
target_include_directories(model-converter-lib SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(model-converter-lib PUBLIC ${CONVERTER_LIB_DEPENDENCIES})
target_compile_options(model-converter-lib PRIVATE ${ML_SDK_MODEL_CONVERTER_COMPILE_OPTIONS})

add_dependencies(model-converter-lib
                MLIRModelConverterPasses
                MLIRVGFDialectIncGen
                MLIRVGFOpsIncGen
                MLIRVGFAttributesIncGen)

add_executable(model-converter "src/main.cpp")
target_link_libraries(model-converter PRIVATE model-converter-lib argparse::argparse)
target_compile_options(model-converter PRIVATE ${ML_SDK_MODEL_CONVERTER_COMPILE_OPTIONS})

mlsdk_generate_version_header(
    TARGET model-converter
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/version.hpp.in"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp"
    DEPENDENCIES
        argparse
        flatbuffers
        LLVM
        MLIR
        VGF
)

add_executable(${MODEL_CONVERTER_NAMESPACE}::model-converter ALIAS model-converter)

install(TARGETS model-converter EXPORT ${MODEL_CONVERTER_PACKAGE_NAME}Config)

if(ML_SDK_BUILD_DOCS OR MODEL_CONVERTER_BUILD_DOCS)
    include(docs/docs.cmake)
endif()
unset(MODEL_CONVERTER_BUILD_DOCS CACHE)

# pytest tests depend on vgf_dump
add_custom_target(model-converter-tests ALL DEPENDS VGF::vgf_dump)

###############################################################################
# Packaging
###############################################################################

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    include(cmake/package.cmake)
    mlsdk_package(PACKAGE_NAME ${MODEL_CONVERTER_PACKAGE_NAME}
        NAMESPACE ${MODEL_CONVERTER_NAMESPACE}
        LICENSES "${CMAKE_CURRENT_LIST_DIR}/LICENSES/Apache-2.0.txt"
            "${CMAKE_CURRENT_LIST_DIR}/LICENSES/LLVM-exception.txt")
endif()

###############################################################################
# MLIR Optimizer
###############################################################################
add_llvm_executable(model-converter-opt src/opt.cpp)

target_include_directories(model-converter-opt PRIVATE src)
target_include_directories(model-converter-opt SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(model-converter-opt PRIVATE
    MLIROptLib
    model-converter-lib)

add_dependencies(model-converter-opt model-converter-lib)

llvm_update_compile_flags(model-converter-opt)
mlir_check_all_link_libraries(model-converter-opt)
