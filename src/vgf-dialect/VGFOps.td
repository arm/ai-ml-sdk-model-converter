/*
* SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/
#ifndef MLIR_VGF_DIALECT_OPS_TD
#define MLIR_VGF_DIALECT_OPS_TD

include "src/vgf-dialect/VGFDialect.td"
include "src/vgf-dialect/VGFEnums.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def InModuleScope : PredOpTrait<"",
  CPred<"$_op.getParentOp()->hasTrait<OpTrait::SymbolTable>()">>;

//===----------------------------------------------------------------------===//
// VGF Dialect Operations
//===----------------------------------------------------------------------===//

class VGF_Op<string mnemonic, list<Trait> traits = []> :
    Op<VGF_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// SequenceOp
//===----------------------------------------------------------------------===//

def SequenceOp : VGF_Op<"sequence",
  [InModuleScope, IsolatedFromAbove, FunctionOpInterface, SymbolTable]> {
  let summary = "TODO";
  let description = [{
    TODO
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);

  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// CallableOpInterface
    Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// SequenceOutputOp
//===----------------------------------------------------------------------===//

def SequenceOutputOp : VGF_Op<"sequence_output",
  [HasParent<"SequenceOp">, Pure, Terminator]> {
  let summary = "TODO";
  let description = [{
    TODO
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

//===----------------------------------------------------------------------===//
// SegmentOp
//===----------------------------------------------------------------------===//

def SegmentOp : VGF_Op<"segment",
  [HasParent<"SequenceOp">, IsolatedFromAbove, FunctionOpInterface, SymbolTable]> {
  let summary = "TODO";
  let description = [{
    TODO
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       SegmentTypeEnumAttr:$segment_type,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);

  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// CallableOpInterface
    Region *getCallableRegion() { return &getBody(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// SegmentRunOp
//===----------------------------------------------------------------------===//

def SegmentRunOp : VGF_Op<"run_segment",
  [HasParent<"SequenceOp">, CallOpInterface]> {
  let summary = "TODO";
  let description = [{
    TODO
  }];

  let arguments = (ins SymbolRefAttr:$segment, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    operand_range getArgOperands() {
      return {operand_begin(), operand_end()};
    }

    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("segment");
    }

    void setCalleeFromCallable(CallInterfaceCallable segment) {
      (*this)->setAttr("segment", segment.get<SymbolRefAttr>());
    }

    ArrayAttr getArgAttrsAttr() { return {}; }
    ArrayAttr getResAttrsAttr() { return {}; }

    void setArgAttrsAttr(ArrayAttr attrs) {}
    void setResAttrsAttr(ArrayAttr attrs) {}

    Attribute removeArgAttrsAttr() { return {}; }
    Attribute removeResAttrsAttr() { return {}; }

  }];

  let assemblyFormat = "$segment (`:``(` $operands^  `)`)? functional-type($operands, results) attr-dict";
}

//===----------------------------------------------------------------------===//
// SegmentOutputOp
//===----------------------------------------------------------------------===//

def SegmentOutputOp : VGF_Op<"segment_output",
  [HasParent<"SegmentOp">, Pure, Terminator]> {
  let summary = "TODO";
  let description = [{
    TODO
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

//===----------------------------------------------------------------------===//
// ShaderPlaceholderOp
//===----------------------------------------------------------------------===//

def ShaderPlaceholderOp : VGF_Op<"shader_placeholder",
  [HasParent<"SegmentOp">]> {

  let summary = "TODO";
  let description = [{
    TODO
  }];

  let arguments = (ins StrAttr:$shader_name,
                       StrAttr:$entry_point,
                       DenseI64ArrayAttr:$input_bindings,
                       DenseI64ArrayAttr:$output_bindings,
                       I64Attr: $input_descriptor_set,
                       I64Attr: $output_descriptor_set,
                       StrArrayAttr:$input_vk_descriptor_types,
                       StrArrayAttr:$output_vk_descriptor_types,
                       StrArrayAttr:$input_vk_formats,
                       StrArrayAttr:$output_vk_formats,
                       DenseI64ArrayAttr:$workgroup_sizes,
                       OptionalAttr<DenseI32ArrayAttr>:$shader_code,
                       Variadic<AnyType>:$operands
  );

  let results = (outs Variadic<AnyType>);

  let assemblyFormat = "`(` $operands `)` attr-dict `:` functional-type($operands, results)";
}

#endif // MLIR_VGF_DIALECT_OPS_TD
