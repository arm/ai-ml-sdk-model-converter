/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#pragma once

#include "llvm/ADT/StringRef.h"

namespace mlir::model_converter_passes {

inline constexpr llvm::StringLiteral graphPartitionIdAttrName = "graph_partition_id";
inline constexpr llvm::StringLiteral graphPartitionLeafNodeAttrName = "graph_partition_leaf_node";
inline constexpr llvm::StringLiteral graphPartitionSequenceOutputIndicesAttrName =
    "graph_partition_sequence_output_indices";
inline constexpr llvm::StringLiteral graphPartitionDeleteAttrName = "graph_partition_delete";

} // namespace mlir::model_converter_passes
