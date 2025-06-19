/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include <map>
#include <string>

namespace mlir::model_converter_passes {

enum class TypeNarrowingMode {
    None = 0,
    Full = 1,
    Partial = 2,
    FullPreserveIO = 3,
};

} // namespace mlir::model_converter_passes
