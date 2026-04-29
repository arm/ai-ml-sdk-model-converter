/*
 * SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "include/custom_op_domains.hpp"
#include "include/passes.hpp"

#include <algorithm>

namespace mlir::model_converter_passes {
#define GEN_PASS_DEF_MODELPARTITIONMARKINGPASS
#include "passes.hpp.inc"
namespace {

class ModelPartitionMarkingPass : public impl::ModelPartitionMarkingPassBase<ModelPartitionMarkingPass> {
  public:
    void runOnOperation() override {
        mlir::ModuleOp moduleOp = getOperation();
        const Type tI32 = IntegerType::get(moduleOp.getContext(), 32);
        int64_t highestPartitionId{-1};
        moduleOp.walk([&tI32, &highestPartitionId](Operation *op) {
            if (llvm::isa<mlir::ModuleOp>(op) || llvm::isa<mlir::func::FuncOp>(op)) {
                return;
            }
            if (llvm::isa<mlir::func::ReturnOp>(op)) {
                for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
                    if (Operation *input = operand.getDefiningOp()) {
                        input->setAttr("graph_partition_leaf_node", BoolAttr::get(op->getContext(), true));
                        if (!op->getAttrOfType<IntegerAttr>("graph_partition_sequence_output_index")) {
                            input->setAttr("graph_partition_sequence_output_index",
                                           IntegerAttr::get(tI32, static_cast<int64_t>(index)));
                        }
                    }
                }
                return;
            }

            op->setAttr("graph_partition_leaf_node", BoolAttr::get(op->getContext(), false));

            int64_t partitionId{-1};
            if (auto customOp = llvm::dyn_cast<mlir::tosa::CustomOp>(op);
                customOp && isVulkanCustomShaderOp(customOp)) {
                partitionId = highestPartitionId + 1;
                op->setAttr("graph_partition_leaf_node", BoolAttr::get(op->getContext(), true));
                for (auto operand : op->getOperands()) {
                    if (Operation *input = operand.getDefiningOp()) {
                        input->setAttr("graph_partition_leaf_node", BoolAttr::get(op->getContext(), true));
                    }
                }
            } else {
                bool setLeafFlags = false;
                // compute the partition id for the current op
                for (auto operand : op->getOperands()) {
                    if (Operation *input = operand.getDefiningOp()) {
                        int64_t parentPartitionId = input->getAttrOfType<IntegerAttr>("graph_partition_id").getInt();
                        if (auto parentCustomOp = llvm::dyn_cast<mlir::tosa::CustomOp>(input);
                            parentCustomOp && isVulkanCustomShaderOp(parentCustomOp)) {
                            parentPartitionId = std::max(highestPartitionId, parentPartitionId + 1);
                        }
                        partitionId = std::max(partitionId, parentPartitionId);
                        if (parentPartitionId < partitionId) {
                            setLeafFlags = true;
                        }
                    }
                }
                if (partitionId < 0) {
                    partitionId = std::max(highestPartitionId, int64_t(0));
                }

                // set leaf node flags on the op inputs if needed
                if (setLeafFlags) {
                    for (auto operand : op->getOperands()) {
                        if (Operation *input = operand.getDefiningOp()) {
                            int64_t parentPartitionId =
                                input->getAttrOfType<IntegerAttr>("graph_partition_id").getInt();
                            if (parentPartitionId < partitionId) {
                                input->setAttr("graph_partition_leaf_node", BoolAttr::get(op->getContext(), true));
                            }
                        }
                    }
                }
            }

            highestPartitionId = std::max(highestPartitionId, partitionId);
            op->setAttr("graph_partition_id", IntegerAttr::get(tI32, partitionId));
        });
    }
};

} // namespace

} // namespace mlir::model_converter_passes
