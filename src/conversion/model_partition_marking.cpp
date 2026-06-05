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

enum class PartitionKind { Graph, Compute };

int64_t assignGraphPartitionAfterCompute(const int64_t candidatePartitionId, const int64_t highestPartitionId,
                                         const PartitionKind highestPartitionKind) {
    if (highestPartitionId < 0) {
        return std::max(candidatePartitionId, int64_t(0));
    }
    const int64_t nextComputeSafePartition =
        highestPartitionKind == PartitionKind::Compute ? highestPartitionId + 1 : highestPartitionId;
    return std::max(candidatePartitionId, nextComputeSafePartition);
}

class ModelPartitionMarkingPass : public impl::ModelPartitionMarkingPassBase<ModelPartitionMarkingPass> {
  public:
    void runOnOperation() override {
        mlir::ModuleOp moduleOp = getOperation();
        const Type tI32 = IntegerType::get(moduleOp.getContext(), 32);
        int64_t highestPartitionId{-1};
        PartitionKind highestPartitionKind = PartitionKind::Graph;
        moduleOp.walk([&tI32, &highestPartitionId, &highestPartitionKind](Operation *op) {
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

            PartitionKind partitionKind = PartitionKind::Graph;
            int64_t partitionId{-1};
            if (auto customOp = llvm::dyn_cast<mlir::tosa::CustomOp>(op);
                customOp && isVulkanCustomShaderOp(customOp)) {
                partitionKind = PartitionKind::Compute;
                partitionId = highestPartitionId + 1;
                op->setAttr("graph_partition_leaf_node", BoolAttr::get(op->getContext(), true));
                for (auto operand : op->getOperands()) {
                    if (Operation *input = operand.getDefiningOp()) {
                        input->setAttr("graph_partition_leaf_node", BoolAttr::get(op->getContext(), true));
                    }
                }
            } else {
                SmallVector<Operation *> inputs;
                // compute the partition id for the current op
                for (auto operand : op->getOperands()) {
                    if (Operation *input = operand.getDefiningOp()) {
                        inputs.push_back(input);
                        int64_t parentPartitionId = input->getAttrOfType<IntegerAttr>("graph_partition_id").getInt();
                        if (auto parentCustomOp = llvm::dyn_cast<mlir::tosa::CustomOp>(input);
                            parentCustomOp && isVulkanCustomShaderOp(parentCustomOp)) {
                            parentPartitionId = assignGraphPartitionAfterCompute(
                                parentPartitionId + 1, highestPartitionId, highestPartitionKind);
                        }
                        partitionId = std::max(partitionId, parentPartitionId);
                    }
                }
                if (partitionId < 0) {
                    // Graph ops without any defining producer in the current sequence stay in a graph-owned
                    // partition. If the most recent partition is compute, start a new graph partition so the
                    // Vulkan custom segment remains single-op.
                    partitionId =
                        assignGraphPartitionAfterCompute(partitionId, highestPartitionId, highestPartitionKind);
                }

                // Set leaf node flags after the current op's final partition is known. This keeps cross-partition
                // operand detection independent of operand order.
                for (Operation *input : inputs) {
                    int64_t parentPartitionId = input->getAttrOfType<IntegerAttr>("graph_partition_id").getInt();
                    if (parentPartitionId < partitionId) {
                        input->setAttr("graph_partition_leaf_node", BoolAttr::get(op->getContext(), true));
                    }
                }
            }

            if (partitionId > highestPartitionId) {
                highestPartitionId = partitionId;
                highestPartitionKind = partitionKind;
            } else if (partitionId == highestPartitionId && partitionKind == PartitionKind::Graph) {
                highestPartitionKind = PartitionKind::Graph;
            }
            op->setAttr("graph_partition_id", IntegerAttr::get(tI32, partitionId));
        });
    }
};

} // namespace

} // namespace mlir::model_converter_passes
