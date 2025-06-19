/*
 * SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {

void AccessDataFor(const mlir::DenseIntOrFPElementsAttr &attr, std::function<void(const char *, size_t)> &&callback);

void AccessDataFor(const mlir::DenseArrayAttr &attr, std::function<void(const char *, size_t)> &&callback);

template <typename AttrType>
void AccessData(const AttrType &attr, std::function<void(const char *, size_t)> &&callback) {
    AccessDataFor(attr, std::move(callback));
}

ShapedType convertShapedType(Type type);

void SplitString(std::string text, const std::string &del, std::vector<std::string> &parts);

} // namespace mlir
