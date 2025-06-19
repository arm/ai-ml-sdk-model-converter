/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "utils.hpp"

namespace mlir {

namespace {
size_t GetSizeOfType(mlir::Type type) { return (type.getIntOrFloatBitWidth() + 7) / 8; }

void UnsplatData(mlir::ArrayRef<char> array, size_t elementSize, size_t numElements,
                 std::function<void(const char *, size_t)> &&callback) {
    size_t targetSize = numElements * elementSize;

    std::vector<char> data(targetSize);
    for (size_t target = 0; target < targetSize; target += elementSize) {
        for (size_t j = 0; j < elementSize; ++j) {
            data[target + j] = array[j];
        }
    }

    callback(data.data(), targetSize);
}
} // namespace

void AccessDataFor(const mlir::DenseIntOrFPElementsAttr &attr, std::function<void(const char *, size_t)> &&callback) {
    mlir::DenseElementsAttr actual = attr;
    if (actual.getElementType().isInteger(48) || actual.getElementType().isInteger(4) ||
        actual.getElementType().isInteger(1)) {
        unsigned width = actual.getElementType().isInteger(48) ? 64 : 8;
        mlir::IntegerType newElementType = IntegerType::get(actual.getContext(), width);
        actual = actual.mapValues(newElementType,
                                  [&](const APInt &value) { return value.sextOrTrunc(newElementType.getWidth()); });
    }

    if (actual.isSplat()) {
        UnsplatData(actual.getRawData(), GetSizeOfType(actual.getElementType()), size_t(actual.getNumElements()),
                    std::move(callback));
    } else {
        callback(actual.getRawData().data(), actual.getRawData().size());
    }
}

void AccessDataFor(const mlir::DenseArrayAttr &attr, std::function<void(const char *, size_t)> &&callback) {
    callback(attr.getRawData().data(), attr.getRawData().size());
}

ShapedType convertShapedType(Type type) {
    ShapedType shapedType = llvm::dyn_cast<ShapedType>(type);
    if (shapedType.getRank() == 0) {
        SmallVector<int64_t> shape = {1};
        return shapedType.cloneWith(shape, shapedType.getElementType());
    }
    return shapedType;
}

void SplitString(std::string text, const std::string &del, std::vector<std::string> &parts) {
    size_t pos = 0;
    while ((pos = text.find(del)) != std::string::npos) {
        parts.emplace_back(text.substr(0, pos));
        text.erase(0, pos + del.length());
    }
    parts.emplace_back(text);
}

} // namespace mlir
