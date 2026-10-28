#
# SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Shared by the native project and the standalone documentation project.
get_filename_component(MODEL_CONVERTER_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(MODEL_CONVERTER_DOCS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(MODEL_CONVERTER_DOCS_DIR "${MODEL_CONVERTER_DOCS_BINARY_DIR}/docs")
list(APPEND CMAKE_MODULE_PATH "${MODEL_CONVERTER_REPO_ROOT}/cmake")

if(MODEL_CONVERTER_DOCS_STANDALONE)
    find_package(Doxygen 1.9.1 REQUIRED)
    find_package(Sphinx REQUIRED)
else()
    find_package(Doxygen)
    find_package(Sphinx)
    if(NOT DOXYGEN_FOUND OR NOT SPHINX_FOUND)
        message(STATUS "Doxygen or Sphinx is not available. Cannot generate documentation.")
        return()
    endif()
    if(CMAKE_CROSSCOMPILING)
        message(WARNING "Cannot build the documentation when cross-compiling. Skipping.")
        return()
    endif()
endif()

set(SPHINX_SRC_DIR_IN "${MODEL_CONVERTER_REPO_ROOT}/docs/source")
set(SPHINX_SRC_DIR "${MODEL_CONVERTER_DOCS_DIR}/in")
set(SPHINX_GEN_DIR "${MODEL_CONVERTER_DOCS_DIR}/generated")
set(SPHINX_SOURCES_DIR "${MODEL_CONVERTER_DOCS_DIR}/sources")
set(DOC_STAGED_FILES)

foreach(SRC_IN README.md CONTRIBUTING.md SECURITY.md
        LICENSES/Apache-2.0.txt LICENSES/LLVM-exception.txt)
    set(DOC_SOURCE_FILE "${SPHINX_GEN_DIR}/${SRC_IN}")
    configure_file("${MODEL_CONVERTER_REPO_ROOT}/${SRC_IN}" "${DOC_SOURCE_FILE}" COPYONLY)
    list(APPEND DOC_STAGED_FILES "${DOC_SOURCE_FILE}")
endforeach()
set(DOC_SOURCE_FILE "${SPHINX_GEN_DIR}/model_converter_help.txt")
configure_file("${MODEL_CONVERTER_REPO_ROOT}/docs/help/model_converter_help.rst"
    "${DOC_SOURCE_FILE}" COPYONLY)
list(APPEND DOC_STAGED_FILES "${DOC_SOURCE_FILE}")

file(GLOB_RECURSE DOC_SRC_FILES CONFIGURE_DEPENDS LIST_DIRECTORIES false
    RELATIVE "${SPHINX_SRC_DIR_IN}" "${SPHINX_SRC_DIR_IN}/*")
foreach(SRC_IN IN LISTS DOC_SRC_FILES)
    set(DOC_SOURCE_FILE "${SPHINX_SRC_DIR}/${SRC_IN}")
    configure_file("${SPHINX_SRC_DIR_IN}/${SRC_IN}" "${DOC_SOURCE_FILE}" COPYONLY)
    list(APPEND DOC_STAGED_FILES "${DOC_SOURCE_FILE}")
endforeach()

# The legacy Doxyfile reads only files directly in src (RECURSIVE=NO).
file(GLOB DOC_CODE_FILES CONFIGURE_DEPENDS LIST_DIRECTORIES false
    RELATIVE "${MODEL_CONVERTER_REPO_ROOT}/src"
    "${MODEL_CONVERTER_REPO_ROOT}/src/*.c"
    "${MODEL_CONVERTER_REPO_ROOT}/src/*.cpp"
    "${MODEL_CONVERTER_REPO_ROOT}/src/*.h"
    "${MODEL_CONVERTER_REPO_ROOT}/src/*.hpp"
    "${MODEL_CONVERTER_REPO_ROOT}/src/*.md"
    "${MODEL_CONVERTER_REPO_ROOT}/src/*.dox")
set(DOC_DOXYGEN_FILES)
foreach(SRC_IN IN LISTS DOC_CODE_FILES)
    set(DOC_SOURCE_FILE "${SPHINX_SOURCES_DIR}/src/${SRC_IN}")
    configure_file("${MODEL_CONVERTER_REPO_ROOT}/src/${SRC_IN}"
        "${DOC_SOURCE_FILE}" COPYONLY)
    list(APPEND DOC_STAGED_FILES "${DOC_SOURCE_FILE}")
    list(APPEND DOC_DOXYGEN_FILES "${DOC_SOURCE_FILE}")
endforeach()

# Manifests make deletions visible to render commands and remove staged leftovers.
set(DOC_STAGED_MANIFEST "${MODEL_CONVERTER_DOCS_DIR}/staged-files.txt")
if(EXISTS "${DOC_STAGED_MANIFEST}")
    file(STRINGS "${DOC_STAGED_MANIFEST}" DOC_PREVIOUS_FILES)
    foreach(DOC_PREVIOUS_FILE IN LISTS DOC_PREVIOUS_FILES)
        cmake_path(IS_PREFIX MODEL_CONVERTER_DOCS_DIR "${DOC_PREVIOUS_FILE}"
            NORMALIZE DOC_IS_STAGED)
        if(DOC_IS_STAGED AND NOT DOC_PREVIOUS_FILE IN_LIST DOC_STAGED_FILES)
            file(REMOVE "${DOC_PREVIOUS_FILE}")
        endif()
    endforeach()
endif()
list(JOIN DOC_STAGED_FILES "\n" DOC_STAGED_CONTENT)
file(CONFIGURE OUTPUT "${DOC_STAGED_MANIFEST}" CONTENT "${DOC_STAGED_CONTENT}\n" @ONLY)
set(DOC_DOXYGEN_MANIFEST "${MODEL_CONVERTER_DOCS_DIR}/doxygen-inputs.txt")
list(JOIN DOC_DOXYGEN_FILES "\n" DOC_DOXYGEN_CONTENT)
file(CONFIGURE OUTPUT "${DOC_DOXYGEN_MANIFEST}" CONTENT "${DOC_DOXYGEN_CONTENT}\n" @ONLY)

include("${MODEL_CONVERTER_REPO_ROOT}/cmake/doxygen.cmake")
include("${MODEL_CONVERTER_REPO_ROOT}/cmake/sphinx.cmake")

add_custom_target(model_converter_doc ALL
    DEPENDS model_converter_doxy_doc model_converter_sphx_doc
    SOURCES "${SPHINX_SRC_DIR}/index.rst")
