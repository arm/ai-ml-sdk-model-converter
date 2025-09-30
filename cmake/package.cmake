#
# SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

macro(mlsdk_package)
    cmake_parse_arguments(ARGS "" "PACKAGE_NAME;NAMESPACE" "LICENSES" ${ARGN})

    set(ALL_LICENSES_CONTENT "")
    foreach(LICENSE ${ARGS_LICENSES})
        file(READ "${LICENSE}" LICENSE_CONTENT)
        string(APPEND ALL_LICENSES_CONTENT "${LICENSE_CONTENT}")
    endforeach()
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/LICENSE" "${ALL_LICENSES_CONTENT}")

    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_BINARY_DIR}/LICENSE")
    set(CPACK_PACKAGE_VERSION "none")

    find_package(Git)

    if(Git_FOUND)
        execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --long --match=[0-9][0-9].[0-9][0-9]
            WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
            RESULT_VARIABLE RESULT
            OUTPUT_VARIABLE GIT_DESCRIBE
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)

        if(RESULT EQUAL "0")
            # Transform output from 24.08-16-g00d8931 to 24.08.16.g00d8931
            string(REGEX MATCH "^([0-9]+).([0-9]+)-([0-9]+)-([a-z0-9]+)" TMP ${GIT_DESCRIBE})
            set(CPACK_PACKAGE_VERSION_MAJOR ${CMAKE_MATCH_1})
            set(CPACK_PACKAGE_VERSION_MINOR ${CMAKE_MATCH_2})
            set(CPACK_PACKAGE_VERSION_PATCH "${CMAKE_MATCH_3}.${CMAKE_MATCH_4}")
            set(CPACK_PACKAGE_VERSION "${CPACK_PACKAGE_VERSION_MAJOR}.${CPACK_PACKAGE_VERSION_MINOR}.${CPACK_PACKAGE_VERSION_PATCH}")
        endif()
    endif()

    if(CMAKE_SYSTEM_PROCESSOR)
        set(CPACK_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}_${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    set(CPACK_VERBATIM_VARIABLES TRUE)

    include(CPack)
    include(CMakePackageConfigHelpers)

    write_basic_package_version_file("${ARGS_PACKAGE_NAME}ConfigVersion.cmake"
        VERSION ${CPACK_PACKAGE_VERSION}
        COMPATIBILITY ExactVersion)

    install(EXPORT ${ARGS_PACKAGE_NAME}Config
            NAMESPACE ${ARGS_NAMESPACE}::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${ARGS_PACKAGE_NAME}")

    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${ARGS_PACKAGE_NAME}ConfigVersion.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${ARGS_PACKAGE_NAME}")
    install(FILES "${PROJECT_SOURCE_DIR}/ai-ml-sdk-model-converter.spdx.json"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/")
endmacro()
