/*
 * SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#include "include/passes.hpp"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::model_converter_passes {
namespace {

//----------------------------------------------------------------------------//
// Pass Definition
//----------------------------------------------------------------------------//
class DenseResourceInlinerPass final : public DenseResourceInlinerPassBase<DenseResourceInlinerPass> {

    void runOnOperation() override {
        auto funcOp = getOperation();
        funcOp.walk([&](tosa::ConstOp constOp) {
            const auto value = constOp.getValues();

            // Check if the value is a DenseResourceElementsAttr
            auto resourceAttr = llvm::dyn_cast<DenseResourceElementsAttr>(value);
            if (!resourceAttr)
                return;

            // Check if there is a memory blob attached to the resource
            AsmResourceBlob *blob = resourceAttr.getRawHandle().getBlob();
            if (!blob)
                return;

            ArrayRef<char> data = blob->getData();
            const auto attrType = constOp.getType();
            bool isSplat = false;
            if (!DenseElementsAttr::isValidRawBuffer(attrType, data, isSplat))
                return;

            auto denseElementsAttr = DenseElementsAttr::getFromRawBuffer(attrType, data);
            constOp->setAttr("values", denseElementsAttr);
        });
    }
};

} // namespace

std::unique_ptr<Pass> createDenseResourceInlinerPass() { return std::make_unique<DenseResourceInlinerPass>(); }

void registerDenseResourceInlinerPass() {
    PassRegistration<DenseResourceInlinerPass>(
        []() -> std::unique_ptr<Pass> { return createDenseResourceInlinerPass(); });
}

} // namespace mlir::model_converter_passes
