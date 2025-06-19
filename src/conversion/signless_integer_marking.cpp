/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "include/passes.hpp"

namespace mlir {
namespace model_converter_passes {
namespace {
class SignlessIntegerMarkingPass : public SignlessIntegerMarkingPassBase<SignlessIntegerMarkingPass> {
  public:
    SignlessIntegerMarkingPass() = default;

    void runOnOperation() override {
        if (SignlessIntegerMarking(getOperation()).failed()) {
            return signalPassFailure();
        }
    }

  private:
    mlir::LogicalResult SignlessIntegerMarking(mlir::func::FuncOp func) {
        MLIRContext *context = &getContext();
        const FunctionType funcType = func.getFunctionType();
        StringRef unsigned_attr = "mlsdk.unsigned_input_output";

        // set the default unsigned_input_output attribute for inputs
        for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
            func.setArgAttr(i, unsigned_attr, BoolAttr::get(context, false));
        }
        // set the default unsigned_input_output attribute for outputs
        for (unsigned i = 0; i < funcType.getNumResults(); ++i) {
            func.setResultAttr(i, unsigned_attr, BoolAttr::get(context, false));
        }

        // find rescale operations attached to input
        func.walk([&](tosa::RescaleOp op) {
            for (auto operand : op->getOperands()) {
                if (operand.getDefiningOp() == nullptr) {
                    if (op.getInputUnsigned()) {
                        if (BlockArgument arg = llvm::dyn_cast<BlockArgument>(operand)) {
                            unsigned inputIndex = arg.getArgNumber();
                            func.setArgAttr(inputIndex, unsigned_attr, BoolAttr::get(context, true));
                        }
                    }
                }
            }
        });

        // find rescale operations attached to output
        func.walk([&](mlir::func::ReturnOp op) {
            for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
                if (Operation *input = operand.getDefiningOp()) {
                    if (auto rescaleOp = llvm::dyn_cast_or_null<tosa::RescaleOp>(input)) {
                        if (rescaleOp.getOutputUnsigned()) {
                            func.setResultAttr(static_cast<uint32_t>(index), unsigned_attr,
                                               BoolAttr::get(context, true));
                        }
                    }
                }
            }
        });

        return mlir::success();
    }
};

} // namespace

std::unique_ptr<Pass> createSignlessIntegerMarkingPass() { return std::make_unique<SignlessIntegerMarkingPass>(); }

void registerSignlessIntegerMarkingPass() {
    PassRegistration<SignlessIntegerMarkingPass>(
        []() -> std::unique_ptr<Pass> { return createSignlessIntegerMarkingPass(); });
}

} // namespace model_converter_passes
} // namespace mlir
