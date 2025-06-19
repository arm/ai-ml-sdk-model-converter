/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "include/passes.hpp"
#include "utils.hpp"

#include <numeric>

namespace mlir {
namespace model_converter_passes {
namespace {

class CheckConstantSparsityPass : public CheckConstantSparsityPassBase<CheckConstantSparsityPass> {
  public:
    void runOnOperation() override {

        mlir::ModuleOp moduleOp = getOperation();

        moduleOp.walk([this](Operation *op) {
            if (!isOpSupported(op))
                return;

            Operation *definingOp = op->getOperand(1).getDefiningOp();
            if (definingOp == nullptr)
                return;

            if (!llvm::isa<tosa::ConstOp>(*definingOp))
                return;

            mlir::DenseIntOrFPElementsAttr value =
                llvm::dyn_cast<DenseIntOrFPElementsAttr>(definingOp->getAttr("values"));

            const auto shape = convertShapedType(op->getOperand(1).getType()).getShape();

            if (shape.size() != 4 || (shape[1] * shape[2]) == 1) {
                return;
            }

            int64_t weightZp = 0;
            auto quantizationInfo = op->getAttr("quantization_info");

            if (quantizationInfo != nullptr) {
                weightZp = llvm::cast<tosa::ConvOpQuantizationAttr>(quantizationInfo).getWeightZp();
            }

            int64_t icDim = shape[shape.size() - 1];
            int64_t otherDim = std::accumulate(shape.begin(), shape.end() - 1, int64_t(1), std::multiplies<int64_t>());

            if (checkSparsity(value, otherDim, icDim, weightZp)) {
                const Type typeId =
                    IntegerType::get(definingOp->getContext(), 32, IntegerType::SignednessSemantics::Signless);
                definingOp->setAttr("constant_2_4_sparse_on_dimension",
                                    IntegerAttr::get(typeId, static_cast<int64_t>(shape.size() - 1)));
            }
        });
    }

  private:
    bool isOpSupported(Operation *op);
    bool checkSparsity(mlir::DenseIntOrFPElementsAttr &value, int64_t otherDim, int64_t icDim, int64_t zp);
    template <typename T> bool checkSparsityLoop(const T *data, int64_t otherDim, int64_t icDim, T zp);
};

bool CheckConstantSparsityPass::isOpSupported(Operation *op) {
    return llvm::isa<tosa::Conv2DOp>(op) || llvm::isa<tosa::TransposeConv2DOp>(op);
}

bool CheckConstantSparsityPass::checkSparsity(mlir::DenseIntOrFPElementsAttr &value, int64_t otherDim, int64_t icDim,
                                              int64_t zp) {

    mlir::Type elementType = value.getElementType();

    if (elementType.isInteger(8) || elementType.isInteger(4)) {
        const int8_t *data = reinterpret_cast<const int8_t *>(value.getRawData().data());
        return checkSparsityLoop(data, otherDim, icDim, static_cast<int8_t>(zp));

    } else if (elementType.isF32()) {
        const uint32_t *data = reinterpret_cast<const uint32_t *>(value.getRawData().data());
        return checkSparsityLoop(data, otherDim, icDim, static_cast<uint32_t>(zp));

    } else if (elementType.isF16()) {
        const uint16_t *data = reinterpret_cast<const uint16_t *>(value.getRawData().data());
        return checkSparsityLoop(data, otherDim, icDim, static_cast<uint16_t>(zp));
    }
    return false;
}

template <typename T>
bool CheckConstantSparsityPass::checkSparsityLoop(const T *data, int64_t otherDim, int64_t icDim, T zp) {
    for (int64_t j = 0; j < otherDim; ++j) {

        for (int64_t i = 0; i < icDim; i += 4) {
            int zeroCount = 0;

            for (int64_t weightIdx = i; weightIdx < (i + 4) && (weightIdx < icDim); ++weightIdx) {

                if (data[icDim * j + weightIdx] == zp)
                    zeroCount++;
            }
            if (zeroCount < 2)
                return false;
        }
    }
    return true;
}

} // namespace

std::unique_ptr<Pass> createCheckConstantSparsityPass() { return std::make_unique<CheckConstantSparsityPass>(); }

void registerCheckConstantSparsityPass() {
    PassRegistration<CheckConstantSparsityPass>(
        []() -> std::unique_ptr<Pass> { return createCheckConstantSparsityPass(); });
}

} // namespace model_converter_passes
} // namespace mlir
