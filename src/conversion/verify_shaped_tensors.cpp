/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "include/passes.hpp"
#include "utils.hpp"

#include "llvm/ADT/STLExtras.h"

#include <memory>

namespace mlir::model_converter_passes {
#define GEN_PASS_DEF_TOSASHAPEDVERIFICATIONPASS
#include "passes.hpp.inc"

namespace {
bool isRealizedShape(ArrayRef<int64_t> shape) {
    return llvm::all_of(shape, [](int64_t dim) { return dim > 0; });
}
} // namespace

class TosaShapedVerificationPass : public impl::TosaShapedVerificationPassBase<TosaShapedVerificationPass> {
  public:
    void runOnOperation() final {
        auto funcOp = getOperation();

        const auto checkType = [&](Type type, Operation *owner) {
            const auto tensor = llvm::dyn_cast<TensorType>(type);
            if (!tensor) {
                return;
            }
            if (!tensor.hasRank()) {
                owner->emitError() << "Unranked tensor is not supported: " << tensor;
                signalPassFailure();
                return;
            }
            if (!isRealizedShape(tensor.getShape())) {
                owner->emitError() << "Tensor dimensions must be statically positive: " << tensor;
                signalPassFailure();
            }
        };

        // Shape validation must not depend on optional input-name metadata.
        // Signatures include declarations and results without SSA users.
        const auto functionType = funcOp.getFunctionType();
        for (Type type : llvm::concat<const Type>(functionType.getInputs(), functionType.getResults())) {
            checkType(type, funcOp);
        }
        funcOp->walk([&](Operation *op) {
            for (Type type : llvm::concat<const Type>(op->getOperandTypes(), op->getResultTypes())) {
                checkType(type, op);
            }
            for (Region &region : op->getRegions()) {
                for (Block &block : region) {
                    for (BlockArgument argument : block.getArguments()) {
                        checkType(argument.getType(), op);
                    }
                }
            }
        });
    }
};

} // namespace mlir::model_converter_passes
