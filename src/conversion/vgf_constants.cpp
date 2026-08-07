/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "include/passes.hpp"
#include "mlir/Conversion/TosaToSPIRVTosa/TosaToSPIRVTosa.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "utils.hpp"
#include "vgf_builder.hpp"

#include <fstream>
#include <map>

using namespace mlsdk::vgflib;

namespace mlir::model_converter_passes {
#define GEN_PASS_DEF_VGFCONSTANTSPASS
#include "passes.hpp.inc"
namespace {

class VGFConstantsPass : public impl::VGFConstantsPassBase<VGFConstantsPass> {
  public:
    VGFConstantsPass() : VGFConstantsPass(std::make_shared<VGFBuilder>()) {}

    explicit VGFConstantsPass(std::shared_ptr<VGFBuilder> VGFBuilder) : _VGFBuilder(std::move(VGFBuilder)) {}

    void runOnOperation() override {
        if (serializeConstants(getOperation()).failed()) {
            llvm::errs() << "Unable to serialize constants\n";
            signalPassFailure();
            return;
        }
    }

  private:
    template <typename T>
    void serializeConstantData(uint32_t graphConstantId, T &attr, ResourceRef resource,
                               int64_t sparsityDimension = -1) {
        mlir::AccessData(attr, [&](const char *data, size_t size) {
            ConstantRef constRef = _VGFBuilder->getEncoder().AddConstant(resource, data, size, sparsityDimension);
            _VGFBuilder->AddConstantBinding(graphConstantId, constRef);
        });
    }

    bool checkIfUnsignedRequired(tosa::ConstOp constOp) {
        auto onlyRescaleOpsWithUnsignedInput = [&constOp](Operation *userOp) {
            if (auto rescaleOp = llvm::dyn_cast_or_null<tosa::RescaleOp>(userOp)) {
                return rescaleOp.getInputUnsigned() && rescaleOp->getOperand(0) == constOp;
            }
            return false;
        };
        return constOp.getType().getElementType().isSignlessInteger() && !constOp->use_empty() &&
               llvm::all_of(constOp->getUsers(), onlyRescaleOpsWithUnsignedInput);
    }

    mlir::LogicalResult serializeConstants(mlir::ModuleOp moduleOp) {

        std::map<uint32_t, tosa::ConstOp> constantsById;
        moduleOp.walk([&constantsById](Operation *op) {
            if (auto constOp = llvm::dyn_cast<tosa::ConstOp>(op)) {
                auto id = constOp->getAttrOfType<IntegerAttr>(tosa::graphARMGraphConstantIdAttrName);
                if (id != nullptr) {
                    constantsById.try_emplace(static_cast<uint32_t>(id.getInt()), constOp);
                }
            }
        });

        for (auto &[id, constOp] : constantsById) {
            const ShapedType type = convertShapedType(constOp.getResult().getType());
            VGFBuilder::VkFormat vkFormat;
            if (VGFBuilder::mlirTypeToVkFormat(type.getElementType(), vkFormat, checkIfUnsignedRequired(constOp))
                    .failed()) {
                return constOp.emitError("unsupported type for tosa.const op: ") << type.getElementType();
            }

            auto format = static_cast<FormatType>(vkFormat);
            ResourceRef resourceRef =
                _VGFBuilder->getEncoder().AddConstantResource(format, flagAnyDynamicDims(type.getShape()), {});

            int64_t sparsityDimension = -1;
            auto attr = constOp->getAttrOfType<IntegerAttr>("constant_2_4_sparse_on_dimension");
            if (attr != nullptr) {
                sparsityDimension = attr.getInt();
            }

            auto attrVal = llvm::dyn_cast<DenseTypedElementsAttr>(constOp.getValuesAttr());
            serializeConstantData(id, attrVal, resourceRef, sparsityDimension);
        }

        return mlir::success();
    }

    std::shared_ptr<VGFBuilder> _VGFBuilder;
};

} // namespace

std::unique_ptr<Pass> createVGFConstantsPass(std::shared_ptr<VGFBuilder> vgfBuilder) {
    return std::make_unique<VGFConstantsPass>(std::move(vgfBuilder));
}

} // namespace mlir::model_converter_passes
