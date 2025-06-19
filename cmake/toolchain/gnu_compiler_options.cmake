#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Compilation warnings
set(ML_SDK_MODEL_CONVERTER_COMPILE_OPTIONS -Werror -Wall -Wextra -Wsign-conversion -Wconversion -Wpedantic)

if(MODEL_CONVERTER_GCC_SANITIZERS)
    message(STATUS "GCC Sanitizers enabled")
    add_compile_options(
        -fsanitize=undefined,address
        -fno-sanitize=alignment
        -fno-sanitize-recover=all
        -fno-sanitize=vptr
    )
    add_link_options(
        -fsanitize=undefined,address
    )
    unset(MODEL_CONVERTER_GCC_SANITIZERS CACHE)
endif()
