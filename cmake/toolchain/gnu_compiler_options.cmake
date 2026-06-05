#
# SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# Compilation warnings
set(ML_SDK_MODEL_CONVERTER_COMPILE_OPTIONS -Werror -Wall -Wextra -Wsign-conversion -Wconversion -Wpedantic)

# GCC can report a false-positive maybe-uninitialized warning from generated
# MLIR operation property hashing when compiling with optimizations.
list(APPEND ML_SDK_MODEL_CONVERTER_COMPILE_OPTIONS -Wno-maybe-uninitialized)
