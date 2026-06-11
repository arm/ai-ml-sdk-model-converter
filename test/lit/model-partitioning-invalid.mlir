//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: not model-converter-opt --model-partition-marking %s 2>&1 | FileCheck %s --check-prefix=MARKING
// RUN: not model-converter-opt --model-partitioning %s 2>&1 | FileCheck %s --check-prefix=PARTITIONING

func.func private @declared(%arg0: tensor<4xi8>) -> tensor<4xi8>

// MARKING: error: model partition marking requires function definitions
// PARTITIONING: error: model partitioning requires function definitions
