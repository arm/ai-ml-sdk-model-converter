#
# SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
include(cmake/doxygen.cmake)
include(cmake/sphinx.cmake)

if(NOT DOXYGEN_FOUND OR NOT SPHINX_FOUND)
  return()
endif()

if(CMAKE_CROSSCOMPILING)
    message(WARNING "Cannot build the documentation when cross-compiling. Skipping.")
    return()
endif()

file(MAKE_DIRECTORY ${SPHINX_GEN_DIR})

# Copy MD files for inclusion into the published docs
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/CONTRIBUTING.md ${SPHINX_GEN_DIR}/CONTRIBUTING.md COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/README.md ${SPHINX_GEN_DIR}/README.md COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/SECURITY.md ${SPHINX_GEN_DIR}/SECURITY.md COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LICENSES/Apache-2.0.txt ${SPHINX_GEN_DIR}/LICENSES/Apache-2.0.txt COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/LICENSES/LLVM-exception.txt ${SPHINX_GEN_DIR}/LICENSES/LLVM-exception.txt COPYONLY)

# Generate a text file with model-converter tool help text
set(MODEL_CONVERTER_ARG_HELP_TXT ${SPHINX_GEN_DIR}/model_converter_help.txt)
add_custom_command(
    OUTPUT "${MODEL_CONVERTER_ARG_HELP_TXT}"
    COMMAND ${CMAKE_COMMAND}
            -Dcmd=$<IF:$<PLATFORM_ID:Windows>,.\\,./>$<TARGET_FILE_NAME:${MODEL_CONVERTER_NAMESPACE}::model-converter>
            -Dargs=--help
            -Dwd=$<TARGET_FILE_DIR:${MODEL_CONVERTER_NAMESPACE}::model-converter>
            -Dout=${MODEL_CONVERTER_ARG_HELP_TXT}
            -P ${CMAKE_CURRENT_LIST_DIR}/redirect-output.cmake
    COMMAND_EXPAND_LISTS
    DEPENDS ${MODEL_CONVERTER_NAMESPACE}::model-converter
    BYPRODUCTS ${MODEL_CONVERTER_ARG_HELP_TXT}
    VERBATIM
    COMMENT "Generating model-converter tool ARGPARSE help documentation"
)

set(DOC_SRC_FILES_FULL_PATHS
    ${SPHINX_GEN_DIR}/CONTRIBUTING.md
    ${SPHINX_GEN_DIR}/README.md
    ${SPHINX_GEN_DIR}/SECURITY.md
    ${MODEL_CONVERTER_ARG_HELP_TXT})

# Set source inputs list
file(GLOB_RECURSE DOC_SRC_FILES CONFIGURE_DEPENDS RELATIVE ${SPHINX_SRC_DIR_IN} ${SPHINX_SRC_DIR_IN}/*)
foreach(SRC_IN IN LISTS DOC_SRC_FILES)
    set(DOC_SOURCE_FILE_IN "${SPHINX_SRC_DIR_IN}/${SRC_IN}")
    set(DOC_SOURCE_FILE "${SPHINX_SRC_DIR}/${SRC_IN}")
    configure_file(${DOC_SOURCE_FILE_IN} ${DOC_SOURCE_FILE} COPYONLY)
    list(APPEND DOC_SRC_FILES_FULL_PATHS ${DOC_SOURCE_FILE})
endforeach()

add_custom_command(
    OUTPUT ${SPHINX_INDEX_HTML}
    DEPENDS ${DOC_SRC_FILES_FULL_PATHS}
    COMMAND ${SPHINX_EXECUTABLE} -b html -W -Dbreathe_projects.MLSDK=${DOXYGEN_XML_GEN} ${SPHINX_SRC_DIR} ${SPHINX_BLD_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation in sphinx"
    VERBATIM
)

# Main target to build the docs
add_custom_target(model_converter_doc ALL DEPENDS model_converter_doxy_doc model_converter_sphx_doc SOURCES "${SPHINX_SRC_DIR}/index.rst")
