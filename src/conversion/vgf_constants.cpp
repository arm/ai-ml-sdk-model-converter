/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "include/passes.hpp"
#include "mlir/Conversion/TosaToSPIRV/ConvertTosaConstants.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "utils.hpp"
#include "vgf_builder.hpp"

#include <fstream>
#include <set>

using namespace mlsdk::vgflib;

namespace mlir {
namespace model_converter_passes {
namespace {

class VGFConstantsPass : public VGFConstantsPassBase<VGFConstantsPass> {
  public:
    explicit VGFConstantsPass(std::shared_ptr<VGFBuilder> VGFBuilder) : _VGFBuilder(std::move(VGFBuilder)) {}

    void runOnOperation() override {
        if (serializeConstants(getOperation()).failed()) {
            llvm::errs() << "Unable to serialize constants\n";
            return signalPassFailure();
        }
    }

  private:
    template <typename T> void serializeConstantData(T &attr, ResourceRef resource, int64_t sparsityDimension = -1) {
        mlir::AccessData(attr, [&](const char *data, size_t size) {
            ConstantRef constRef = _VGFBuilder->getEncoder()->AddConstant(resource, data, size, sparsityDimension);
            _VGFBuilder->AddConstantRef(constRef);
        });
    }

    bool checkIfUnsignedRequired(tosa::ConstOp constOp) {
        auto onlyRescaleOpsWithUnsignedInput = [&constOp](Operation *userOp) {
            if (auto rescaleOp = llvm::dyn_cast_or_null<tosa::RescaleOp>(userOp)) {
                return rescaleOp.getInputUnsigned() && rescaleOp->getOperand(0) == constOp;
            }
            return false;
        };
        return constOp.getType().getElementType().isSignlessInteger() &&
               llvm::all_of(constOp->getUsers(), onlyRescaleOpsWithUnsignedInput);
    }

    mlir::LogicalResult serializeConstants(mlir::ModuleOp moduleOp) {

        std::set<uint32_t> processedGraphConstants;
        WalkResult walkResult = moduleOp.walk([this, &processedGraphConstants](Operation *op) {
            if (auto constOp = llvm::dyn_cast<tosa::ConstOp>(op)) {
                const auto id = tosa::getGraphIdForConst(constOp);
                if (id.has_value()) {
                    if (processedGraphConstants.insert(id.value()).second) {
                        const ShapedType type = convertShapedType(constOp.getResult().getType());
                        VGFBuilder::VkFormat vkFormat;
                        if (_VGFBuilder
                                ->mlirTypeToVkFormat(type.getElementType(), vkFormat, checkIfUnsignedRequired(constOp))
                                .failed()) {
                            llvm::errs() << "Unsupported type for tosa.const op at " << op->getLoc() << "\n";
                            return WalkResult::interrupt();
                        }

                        auto format = static_cast<FormatType>(vkFormat);
                        ResourceRef resourceRef =
                            _VGFBuilder->getEncoder()->AddConstantResource(format, type.getShape(), {});

                        int64_t sparsityDimension = -1;
                        auto attr = op->getAttrOfType<IntegerAttr>("constant_2_4_sparse_on_dimension");
                        if (attr != nullptr)
                            sparsityDimension = attr.getInt();

                        auto attrVal = llvm::dyn_cast<DenseIntOrFPElementsAttr>(constOp.getValuesAttr());
                        serializeConstantData(attrVal, resourceRef, sparsityDimension);
                    }
                }
            }
            return WalkResult::advance();
        });

        if (walkResult.wasInterrupted()) {
            return mlir::failure();
        }

        return mlir::success();
    }

    std::shared_ptr<VGFBuilder> _VGFBuilder;
};

} // namespace

std::unique_ptr<Pass> createVGFConstantsPass(std::shared_ptr<VGFBuilder> VGFBuilder) {
    return std::make_unique<VGFConstantsPass>(VGFBuilder);
}

void registerVGFConstantsPass() {
    PassRegistration<VGFConstantsPass>(
        []() -> std::unique_ptr<Pass> { return createVGFConstantsPass(std::make_shared<VGFBuilder>()); });
}

} // namespace model_converter_passes
} // namespace mlir
