/*
 * SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#include "VGFDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

// Implementations
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#include "VGFDialect.cpp.inc"
#include "VGFEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// VGF Dialect
//===----------------------------------------------------------------------===//

namespace mlir::vgf {

void VGFDialect::initialize() {
    addAttributes<
#define GET_ATTRDEF_LIST
#include "VGFAttrs.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "VGFOps.cpp.inc"
        >();
}

//===----------------------------------------------------------------------===//
// VGF Operations
//===----------------------------------------------------------------------===//

using namespace function_interface_impl;

auto funcTypeBuilder = [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> resTypes, VariadicFlag,
                          std::string &) { return builder.getFunctionType(argTypes, resTypes); };

//===----------------------------------------------------------------------===//
// SequenceOp
//===----------------------------------------------------------------------===//

ParseResult SequenceOp::parse(OpAsmParser &opAsmParser, OperationState &opState) {
    return parseFunctionOp(opAsmParser, opState, false, getFunctionTypeAttrName(opState.name), funcTypeBuilder,
                           getArgAttrsAttrName(opState.name), getResAttrsAttrName(opState.name));
}

void SequenceOp::print(OpAsmPrinter &opAsmPointer) {
    printFunctionOp(opAsmPointer, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
                    getResAttrsAttrName());
}

//===----------------------------------------------------------------------===//
// SegmentOp
//===----------------------------------------------------------------------===//

ParseResult SegmentOp::parse(OpAsmParser &opAsmParser, OperationState &opState) {
    return parseFunctionOp(opAsmParser, opState, false, getFunctionTypeAttrName(opState.name), funcTypeBuilder,
                           getArgAttrsAttrName(opState.name), getResAttrsAttrName(opState.name));
}

void SegmentOp::print(OpAsmPrinter &opAsmPointer) {
    printFunctionOp(opAsmPointer, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
                    getResAttrsAttrName());
}

} // namespace mlir::vgf

#define GET_ATTRDEF_CLASSES
#include "VGFAttrs.cpp.inc"

#define GET_OP_CLASSES
#include "VGFOps.cpp.inc"
