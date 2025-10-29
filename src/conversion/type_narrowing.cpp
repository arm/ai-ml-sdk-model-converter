/*
 * SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#include "include/type_narrowing.hpp"
#include "include/passes.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::model_converter_passes {
namespace {

// Reduce the precision of the accumulator type attribute of a given operator
//
// Handles the following down-sizing:
//  * f32 -> f16
//
// e.g.
//   %0 = tosa.avg_pool2d %arg0 {acc_type = f32,
//                               kernel = array<i64: 2, 2>,
//                               pad = array<i64: 0, 1, 0, 1>,
//                               stride = array<i64: 1, 1>}
//      : (tensor<1x7x7x9xf16>) -> tensor<1x7x7x9xf16>
//
// will be transformed into:
//   %0 = tosa.avg_pool2d %arg0 {acc_type = f16,
//                               kernel = array<i64: 2, 2>,
//                               pad = array<i64: 0, 1, 0, 1>,
//                               stride = array<i64: 1, 1>}
//      : (tensor<1x7x7x9xf16>) -> tensor<1x7x7x9xf16>
//
template <typename OpTy, typename OpAdaptor> struct ReduceFloatAccTypePattern : public OpConversionPattern<OpTy> {
    using OpConversionPattern<OpTy>::OpConversionPattern;

    LogicalResult matchAndRewrite(OpTy op, OpAdaptor, ConversionPatternRewriter &rewriter) const override {
        const TypeConverter *converter = this->getTypeConverter();
        rewriter.modifyOpInPlace(op, [&]() { op.setAccType(converter->convertType(op.getAccType())); });
        return success();
    }
};

struct FuncSignatureConvert : public OpConversionPattern<func::FuncOp> {
    using OpConversionPattern<func::FuncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(func::FuncOp funcOp, func::FuncOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {

        std::string newName = adaptor.getSymName().str();
        FunctionType funcType = adaptor.getFunctionType();
        ArrayAttr argAttrs = adaptor.getArgAttrsAttr();
        ArrayAttr resAttrs = adaptor.getResAttrsAttr();
        StringAttr visibilityAttr = adaptor.getSymVisibilityAttr();

        TypeConverter::SignatureConversion signatureConverter(funcType.getNumInputs());
        SmallVector<Type, 2> newResults;
        if (failed(getTypeConverter()->convertSignatureArgs(funcType.getInputs(), signatureConverter))) {
            return funcOp.emitError("failed to convert function input types");
        }
        if (failed(getTypeConverter()->convertTypes(funcType.getResults(), newResults))) {
            return funcOp.emitError("failed to convert function input results");
        }

        auto newFuncType = FunctionType::get(rewriter.getContext(), signatureConverter.getConvertedTypes(), newResults);

        func::FuncOp newFuncOp =
            func::FuncOp::create(rewriter, funcOp.getLoc(), newName, newFuncType, visibilityAttr, argAttrs, resAttrs);

        newFuncOp->setAttrs(adaptor.getAttributes());

        rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(), newFuncOp.end());
        if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *getTypeConverter(), &signatureConverter))) {
            return funcOp.emitError("failed to convert function regions");
        }

        rewriter.eraseOp(funcOp);
        return success();
    }
};

struct TosaOpConvert : public OpInterfaceConversionPattern<tosa::TosaOp> {
    using OpInterfaceConversionPattern<tosa::TosaOp>::OpInterfaceConversionPattern;

    TosaOpConvert(TypeConverter &typeConverter, MLIRContext *ctx)
        : OpInterfaceConversionPattern(typeConverter, ctx, /*benefit=*/1) {}

    LogicalResult matchAndRewrite(tosa::TosaOp tosaOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {

        if (llvm::isa<tosa::CastOp>(tosaOp)) {
            return rewriter.notifyMatchFailure(tosaOp, "Operation of type `tosa.CastOp` should not be modified.");
        }

        SmallVector<Type, 4> newResultTypes;
        if (failed(getTypeConverter()->convertTypes(tosaOp->getResultTypes(), newResultTypes)))
            return failure();

        OperationState state(tosaOp.getLoc(), tosaOp->getName());
        state.addOperands(operands);
        state.addTypes(newResultTypes);
        state.addAttributes(tosaOp->getAttrs());

        Operation *newOp = rewriter.create(state);

        rewriter.replaceOp(tosaOp, newOp->getResults());

        return success();
    }
};

// Remaps the contents of a `tosa::ConstOp` if result_type != attr_type
// Note: Better done with a folder/canonicalizer upstream in the future
struct RemapConstAttrsPattern : OpConversionPattern<tosa::ConstOp> {
    using OpConversionPattern<tosa::ConstOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(tosa::ConstOp constOp, tosa::ConstOpAdaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Type valueType = constOp.getValues().getType();
        if (getTypeConverter()->isLegal(constOp.getOutput().getType()) && getTypeConverter()->isLegal(valueType)) {
            return rewriter.notifyMatchFailure(constOp, "Result matches attr type");
        }

        auto constAttr = llvm::dyn_cast<DenseElementsAttr>(constOp.getValuesAttr());
        if (!constAttr) {
            return rewriter.notifyMatchFailure(constOp, "Not dense attribute");
        }

        // Convert to smaller bit-width
        Type convertedType = getTypeConverter()->convertType(valueType);
        Type elementType = getElementTypeOrSelf(convertedType);
        DenseElementsAttr newConstAttr = constAttr.mapValues(elementType, [&](APFloat val) {
            bool losesInfo = false;
            val.convert(APFloat::IEEEhalf(), APFloat::rmNearestTiesToEven, &losesInfo);
            return val.bitcastToAPInt();
        });

        rewriter.modifyOpInPlace(constOp, [&]() { constOp.setValuesAttr(newConstAttr); });
        return success();
    }
};

struct FuncReturnConvert : public OpConversionPattern<func::ReturnOp> {
    using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(func::ReturnOp retOp, func::ReturnOpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {

        rewriter.replaceOpWithNewOp<func::ReturnOp>(retOp, adaptor.getOperands());
        return success();
    }
};

//----------------------------------------------------------------------------//
// Pass Definition
//----------------------------------------------------------------------------//

template <typename OpTy, typename OpTyAdaptor>
void addReduceAccTypePattern(ConversionTarget &target, RewritePatternSet &patterns, mlir::TypeConverter &typeConverter,
                             MLIRContext *ctx) {
    target.addDynamicallyLegalOp<OpTy>(
        [&](OpTy op) { return typeConverter.isLegal(op.getAccType()) && typeConverter.isLegal(op.getOperation()); });
    patterns.add<ReduceFloatAccTypePattern<OpTy, OpTyAdaptor>>(typeConverter, ctx);
}

class TypeNarrowingPass final : public TypeNarrowingPassBase<TypeNarrowingPass> {
    using TypeNarrowingPassBase::TypeNarrowingPassBase;

  public:
    explicit TypeNarrowingPass(const TypeNarrowingPassOptions &options) { mode = options.mode; }

    void runOnOperation() override {
        MLIRContext *ctx = &getContext();
        mlir::Operation *op = getOperation();

        TypeConverter typeConverter;
        typeConverter.addConversion([&](FloatType) { return Float16Type::get(ctx); });
        typeConverter.addConversion([&](TensorType type) { return type.clone(Float16Type::get(ctx)); });

        typeConverter.addSourceMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
            return tosa::CastOp::create(builder, loc, type, inputs);
        });
        typeConverter.addTargetMaterialization([](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
            return tosa::CastOp::create(builder, loc, type, inputs);
        });

        ConversionTarget target(*ctx);

        target.addDynamicallyLegalDialect<tosa::TosaDialect>([&](Operation *tosaOp) {
            return !llvm::isa<tosa::ConstOp>(tosaOp) && !llvm::isa<tosa::CastOp>(tosaOp) &&
                   typeConverter.isLegal(tosaOp);
        });

        target.addDynamicallyLegalOp<tosa::ConstOp>([&](tosa::ConstOp constOp) {
            return typeConverter.isLegal(constOp.getValues().getType()) &&
                   typeConverter.isLegal(constOp.getOperation());
        });

        target.addDynamicallyLegalOp<tosa::CastOp>([&](tosa::CastOp castOp) {
            return (typeConverter.isLegal(castOp.getInput().getType()) ||
                    typeConverter.isLegal(castOp.getResult().getType())) &&
                   castOp.getInput().getType() != castOp.getResult().getType();
        });

        RewritePatternSet convPatterns(ctx);
        convPatterns.add<TosaOpConvert>(typeConverter, ctx);
        convPatterns.add<RemapConstAttrsPattern>(typeConverter, ctx);

        if (mode != TypeNarrowingMode::FullPreserveIO) {

            target.addDynamicallyLegalOp<func::FuncOp>(
                [&](func::FuncOp funcOp) { return typeConverter.isSignatureLegal(funcOp.getFunctionType()); });

            target.addDynamicallyLegalOp<func::ReturnOp>(
                [&](func::ReturnOp retOp) { return typeConverter.isLegal(retOp); });

            convPatterns.add<FuncSignatureConvert>(typeConverter, ctx);
            convPatterns.add<FuncReturnConvert>(typeConverter, ctx);
        }

        if (mode != TypeNarrowingMode::Partial) {
            addReduceAccTypePattern<tosa::AvgPool2dOp, tosa::AvgPool2dOpAdaptor>(target, convPatterns, typeConverter,
                                                                                 ctx);
            addReduceAccTypePattern<tosa::Conv2DOp, tosa::Conv2DOpAdaptor>(target, convPatterns, typeConverter, ctx);
            addReduceAccTypePattern<tosa::Conv3DOp, tosa::Conv3DOpAdaptor>(target, convPatterns, typeConverter, ctx);
            addReduceAccTypePattern<tosa::DepthwiseConv2DOp, tosa::DepthwiseConv2DOpAdaptor>(target, convPatterns,
                                                                                             typeConverter, ctx);
            addReduceAccTypePattern<tosa::TransposeConv2DOp, tosa::TransposeConv2DOpAdaptor>(target, convPatterns,
                                                                                             typeConverter, ctx);
        }

        if (failed(applyPartialConversion(op, target, std::move(convPatterns)))) {
            return signalPassFailure();
        }
    }
};

} // namespace

std::unique_ptr<Pass> createTypeNarrowingPass() { return std::make_unique<TypeNarrowingPass>(); }

std::unique_ptr<Pass> createTypeNarrowingPass(TypeNarrowingPassOptions options) {
    return std::make_unique<TypeNarrowingPass>(options);
}

void registerTypeNarrowingPass() {
    PassRegistration<TypeNarrowingPass>([]() -> std::unique_ptr<Pass> { return createTypeNarrowingPass(); });
}

} // namespace mlir::model_converter_passes
