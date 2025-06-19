/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "include/passes.hpp"
#include "utils.hpp"

#include <memory>

namespace mlir {
namespace model_converter_passes {

namespace {
bool isRealizedShape(ArrayRef<int64_t> shape) {
    return llvm::all_of(shape, [](int64_t dim) { return dim > 0; });
}
} // namespace

class TosaShapedVerificationPass : public TosaShapedVerificationPassBase<TosaShapedVerificationPass> {
  public:
    void runOnOperation() final {
        auto funcOp = getOperation();

        auto tfEntryFunctionAttr = funcOp->getAttrDictionary().get("tf.entry_function");
        if (!tfEntryFunctionAttr) {
            llvm::errs() << "Can't identify input tensors without entry functions.\n";
            return signalPassFailure();
        }

        auto tfEntryFunctionDictionaryAttr = llvm::dyn_cast<DictionaryAttr>(tfEntryFunctionAttr);
        if (!tfEntryFunctionDictionaryAttr) {
            llvm::errs() << "\"entry_function\" is not a dictionary attribute.\n";
            return signalPassFailure();
        }

        auto inputsAttr = tfEntryFunctionDictionaryAttr.get("inputs");
        if (!inputsAttr) {
            llvm::outs() << "Function has no \"inputs\" attribute. No shape verification is needed.\n";
            return;
        }

        auto inputsStringAttr = llvm::dyn_cast<StringAttr>(inputsAttr);
        if (!inputsStringAttr) {
            llvm::errs() << "\"inputs\" is not a string attribute.\n";
            return signalPassFailure();
        }

        llvm::SmallVector<llvm::StringRef> inputNames;
        inputsStringAttr.strref().split(inputNames, ",", -1, false);

        FunctionType funcTy = funcOp.getFunctionType();
        bool dynamicInputs = false;
        for (auto [inputName, inputType] : llvm::zip_equal(inputNames, funcTy.getInputs())) {
            auto tensorType = llvm::dyn_cast<TensorType>(inputType);
            if (!tensorType) {
                continue;
            }
            if (!isRealizedShape(tensorType.getShape())) {
                llvm::errs() << "Input tensor \'" << inputName << "\' is dynamic: " << tensorType << "\n";
                dynamicInputs = true;
            } else if (!tensorType.hasRank()) {
                llvm::errs() << "Input tensor \'" << inputName << "\' is unranked: " << tensorType << "\n";
                dynamicInputs = true;
            }
        }

        std::vector<std::string> dynamicOps;
        funcOp->walk([&](Operation *op) {
            for (Value operand : op->getOperands()) {
                auto type = operand.getType();
                TensorType tensorType;
                if (!(tensorType = llvm::dyn_cast<TensorType>(type))) {
                    continue;
                }
                // Verify tensor has rank
                if (!tensorType.hasRank()) {
                    std::string _str;
                    llvm::raw_string_ostream _stream(_str);
                    _stream << "(unranked) " << tensorType << " in: " << operand;
                    dynamicOps.push_back(_str);
                    return;
                }

                // Verify all elements > 0
                ArrayRef<int64_t> shape = tensorType.getShape();
                if (!isRealizedShape(shape)) {
                    std::string _str;
                    llvm::raw_string_ostream _stream(_str);
                    _stream << "(dynamic) " << tensorType << " in: " << operand;
                    dynamicOps.push_back(_str);
                    return;
                }
            }
        });

        if (dynamicInputs) {
            llvm::errs() << "There was found a further " << dynamicOps.size() << " dynamic tensors in the graph.\n";
            return signalPassFailure();
        }

        if (dynamicOps.size()) {
            llvm::errs() << "Dynamic shapes found in the graph, "
                         << "but none of the inputs were dynamic:\n";
            constexpr size_t showNum = 25;
            constexpr size_t maxChar = 300;
            for (size_t i = 0; i < dynamicOps.size() && i < showNum; i++) {
                llvm::errs() << "\t" << i << "/" << dynamicOps.size() << ": \""
                             << (dynamicOps[i].size() < maxChar
                                     ? dynamicOps[i]
                                     : std::string(dynamicOps[i].substr(0, maxChar)) + "... (Another " +
                                           std::to_string(dynamicOps[i].size() - maxChar) + " chars.)")
                             << "\"\n";
            }
            if (showNum <= dynamicOps.size()) {
                llvm::errs() << "... (Another " << (dynamicOps.size() - showNum) << " entries hidden for brevity.)\n";
            }
            llvm::errs() << "Hint: The '--shape-override' flag forces shape inference from inputs.\n";
        }
    }
};

std::unique_ptr<Pass> createTosaShapedVerificationPass() { return std::make_unique<TosaShapedVerificationPass>(); }

void registerTosaShapedVerificationPass() {
    PassRegistration<TosaShapedVerificationPass>(
        []() -> std::unique_ptr<Pass> { return createTosaShapedVerificationPass(); });
}

} // namespace model_converter_passes
} // namespace mlir
