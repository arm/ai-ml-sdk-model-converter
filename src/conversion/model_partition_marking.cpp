/*
 * SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "include/custom_op_domains.hpp"
#include "include/passes.hpp"
#include "model_partition_attrs.hpp"

#include <algorithm>
#include <optional>

namespace mlir::model_converter_passes {
#define GEN_PASS_DEF_MODELPARTITIONMARKINGPASS
#include "passes.hpp.inc"
namespace {

enum class PartitionKind { Graph, Compute };

bool isCompileTimeTosaConstant(Operation *op) { return llvm::isa_and_nonnull<tosa::ConstOp, tosa::ConstShapeOp>(op); }

bool isVulkanCustomShaderOperation(Operation *op) {
    auto customOp = llvm::dyn_cast_or_null<tosa::CustomOp>(op);
    return customOp && isVulkanCustomShaderOp(customOp);
}

int64_t assignGraphPartitionAfterCompute(const int64_t candidatePartitionId, const int64_t highestPartitionId,
                                         const PartitionKind highestPartitionKind) {
    if (highestPartitionId < 0) {
        return std::max(candidatePartitionId, int64_t(0));
    }
    const int64_t nextComputeSafePartition =
        highestPartitionKind == PartitionKind::Compute ? highestPartitionId + 1 : highestPartitionId;
    return std::max(candidatePartitionId, nextComputeSafePartition);
}

bool hasReturnUse(Operation *op) {
    return llvm::any_of(op->getUsers(), [](Operation *user) { return llvm::isa<func::ReturnOp>(user); });
}

class PartitionPlan {
  public:
    void plan(func::FuncOp funcOp) {
        collectOps(funcOp);
        assignRuntimePartitions();
        placeCompileTimeConstants();
        deriveLeafAttrs(funcOp);
        writeAttrs(funcOp);
    }

  private:
    void collectOps(func::FuncOp funcOp) {
        llvm::transform(funcOp.getBody().front().without_terminator(), std::back_inserter(ops),
                        [](Operation &op) { return &op; });
    }

    void assignRuntimePartitions() {
        int64_t highestPartitionId = -1;
        PartitionKind highestPartitionKind = PartitionKind::Graph;

        auto updateHighestPartition = [&](int64_t partitionId, PartitionKind kind) {
            if (partitionId > highestPartitionId) {
                highestPartitionId = partitionId;
                highestPartitionKind = kind;
            } else if (partitionId == highestPartitionId && kind == PartitionKind::Graph) {
                highestPartitionKind = PartitionKind::Graph;
            }
        };

        for (Operation *op : ops) {
            if (isCompileTimeTosaConstant(op)) {
                continue;
            }

            if (isVulkanCustomShaderOperation(op)) {
                int64_t partitionId = highestPartitionId + 1;
                recordPartition(op, partitionId, PartitionKind::Compute);
                updateHighestPartition(partitionId, PartitionKind::Compute);
                continue;
            }

            int64_t partitionId = -1;
            for (Value operand : op->getOperands()) {
                Operation *defOp = operand.getDefiningOp();
                if (!defOp || isCompileTimeTosaConstant(defOp)) {
                    continue;
                }

                auto parentPartitionIt = partitionByOp.find(defOp);
                if (parentPartitionIt == partitionByOp.end()) {
                    continue;
                }

                int64_t parentPartitionId = parentPartitionIt->second;
                if (isVulkanCustomShaderOperation(defOp)) {
                    parentPartitionId = assignGraphPartitionAfterCompute(parentPartitionId + 1, highestPartitionId,
                                                                         highestPartitionKind);
                }
                partitionId = std::max(partitionId, parentPartitionId);
            }

            if (partitionId < 0) {
                partitionId = assignGraphPartitionAfterCompute(partitionId, highestPartitionId, highestPartitionKind);
            }

            recordPartition(op, partitionId, PartitionKind::Graph);
            updateHighestPartition(partitionId, PartitionKind::Graph);
        }
    }

    void placeCompileTimeConstants() {
        for (Operation *op : ops) {
            if (!isCompileTimeTosaConstant(op)) {
                continue;
            }

            const std::optional<int64_t> graphPartition = getFirstGraphUserPartition(op);
            const std::optional<int64_t> computePartition = getFirstComputeUserPartition(op);

            if (computePartition && (!graphPartition || computePartition.value() < graphPartition.value())) {
                int64_t constantPartition = ensureGraphPartitionBefore(computePartition.value());
                recordPartition(op, constantPartition, PartitionKind::Graph);
                continue;
            }

            if (graphPartition) {
                recordPartition(op, graphPartition.value(), PartitionKind::Graph);
                continue;
            }

            if (computePartition) {
                int64_t insertedGraphPartition = ensureGraphPartitionBefore(computePartition.value());
                recordPartition(op, insertedGraphPartition, PartitionKind::Graph);
                continue;
            }

            if (hasReturnUse(op)) {
                recordPartition(op, ensureTrailingGraphPartition(), PartitionKind::Graph);
                continue;
            }

            if (std::optional<int64_t> lastGraphPartition = getLastPartitionOfKind(PartitionKind::Graph)) {
                recordPartition(op, lastGraphPartition.value(), PartitionKind::Graph);
                continue;
            }

            recordPartition(op, ensureTrailingGraphPartition(), PartitionKind::Graph);
        }
    }

    void deriveLeafAttrs(func::FuncOp funcOp) {
        for (Operation *op : ops) {
            int64_t partitionId = partitionByOp.lookup(op);
            if (isVulkanCustomShaderOperation(op)) {
                leafOps.insert(op);
            }

            for (Value result : op->getResults()) {
                for (Operation *user : result.getUsers()) {
                    if (auto returnOp = llvm::dyn_cast<func::ReturnOp>(user)) {
                        leafOps.insert(op);
                        for (auto [index, operand] : llvm::enumerate(returnOp.getOperands())) {
                            if (operand == result && !sequenceOutputIndexByValue.contains(result)) {
                                sequenceOutputIndexByValue[result] = static_cast<int64_t>(index);
                            }
                        }
                        continue;
                    }

                    auto userPartitionIt = partitionByOp.find(user);
                    if (userPartitionIt == partitionByOp.end() || userPartitionIt->second == partitionId) {
                        continue;
                    }

                    if (!isRematerializableGraphUse(op, user)) {
                        leafOps.insert(op);
                    }
                }
            }
        }

        Operation *terminator = funcOp.getBody().front().getTerminator();
        for (auto [index, operand] : llvm::enumerate(terminator->getOperands())) {
            if (Operation *defOp = operand.getDefiningOp()) {
                leafOps.insert(defOp);
                if (!sequenceOutputIndexByValue.contains(operand)) {
                    sequenceOutputIndexByValue[operand] = static_cast<int64_t>(index);
                }
            }
        }
    }

    void writeAttrs(func::FuncOp funcOp) {
        Type partitionIdType = IntegerType::get(funcOp.getContext(), 32);
        for (Operation *op : ops) {
            op->setAttr(graphPartitionIdAttrName, IntegerAttr::get(partitionIdType, partitionByOp.lookup(op)));
            op->setAttr(graphPartitionLeafNodeAttrName, BoolAttr::get(op->getContext(), leafOps.contains(op)));
            op->removeAttr(graphPartitionSequenceOutputIndicesAttrName);

            SmallVector<int64_t> outputIndices(op->getNumResults(), -1);
            bool hasOutputIndex = false;
            for (auto [resultNumber, result] : llvm::enumerate(op->getResults())) {
                auto outputIndexIt = sequenceOutputIndexByValue.find(result);
                if (outputIndexIt == sequenceOutputIndexByValue.end()) {
                    continue;
                }
                outputIndices[resultNumber] = outputIndexIt->second;
                hasOutputIndex = true;
            }

            if (!hasOutputIndex) {
                continue;
            }

            op->setAttr(graphPartitionSequenceOutputIndicesAttrName,
                        DenseI64ArrayAttr::get(op->getContext(), outputIndices));
        }
    }

    void recordPartition(Operation *op, int64_t partitionId, PartitionKind kind) {
        partitionByOp[op] = partitionId;
        partitionKinds.try_emplace(partitionId, kind);
    }

    std::optional<int64_t> getFirstGraphUserPartition(Operation *op) const {
        return getFirstUserPartition(op, [&](Operation *user, int64_t userPartitionId) {
            return !isVulkanCustomShaderOperation(user) && !llvm::isa<func::ReturnOp>(user) &&
                   partitionKinds.lookup(userPartitionId) == PartitionKind::Graph;
        });
    }

    std::optional<int64_t> getFirstComputeUserPartition(Operation *op) const {
        return getFirstUserPartition(op, [](Operation *user, int64_t) { return isVulkanCustomShaderOperation(user); });
    }

    template <typename Predicate>
    std::optional<int64_t> getFirstUserPartition(Operation *op, Predicate predicate) const {
        SmallVector<int64_t> userPartitionIds;
        for (Operation *user : op->getUsers()) {
            auto userPartitionIt = partitionByOp.find(user);
            if (userPartitionIt != partitionByOp.end() && predicate(user, userPartitionIt->second)) {
                userPartitionIds.push_back(userPartitionIt->second);
            }
        }

        if (userPartitionIds.empty()) {
            return std::nullopt;
        }
        return *llvm::min_element(userPartitionIds);
    }

    bool isRematerializableGraphUse(Operation *op, Operation *user) const {
        if (!isCompileTimeTosaConstant(op) || llvm::isa<func::ReturnOp>(user) || isVulkanCustomShaderOperation(user)) {
            return false;
        }

        auto userPartitionIt = partitionByOp.find(user);
        return userPartitionIt != partitionByOp.end() &&
               partitionKinds.lookup(userPartitionIt->second) == PartitionKind::Graph;
    }

    std::optional<int64_t> getLastPartitionOfKind(PartitionKind kind) const {
        SmallVector<int64_t> partitionIds;
        for (const auto &[candidateId, candidateKind] : partitionKinds) {
            if (candidateKind == kind) {
                partitionIds.push_back(candidateId);
            }
        }

        if (partitionIds.empty()) {
            return std::nullopt;
        }
        return *llvm::max_element(partitionIds);
    }

    int64_t ensureGraphPartitionBefore(int64_t partitionId) {
        if (partitionId > 0) {
            auto previousKind = partitionKinds.find(partitionId - 1);
            if (previousKind != partitionKinds.end() && previousKind->second == PartitionKind::Graph) {
                return partitionId - 1;
            }
        }

        shiftPartitionsFrom(partitionId);
        partitionKinds[partitionId] = PartitionKind::Graph;
        return partitionId;
    }

    int64_t ensureTrailingGraphPartition() {
        int64_t highestPartitionId = getHighestPartitionId();
        if (highestPartitionId < 0) {
            partitionKinds[0] = PartitionKind::Graph;
            return 0;
        }

        if (partitionKinds[highestPartitionId] == PartitionKind::Graph) {
            return highestPartitionId;
        }

        partitionKinds[highestPartitionId + 1] = PartitionKind::Graph;
        return highestPartitionId + 1;
    }

    int64_t getHighestPartitionId() const {
        if (partitionKinds.empty()) {
            return -1;
        }

        return llvm::max_element(partitionKinds, [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; })
            ->first;
    }

    void shiftPartitionsFrom(int64_t firstPartitionToShift) {
        for (auto &[_, partitionId] : partitionByOp) {
            if (partitionId >= firstPartitionToShift) {
                ++partitionId;
            }
        }

        DenseMap<int64_t, PartitionKind> shiftedPartitionKinds;
        for (const auto &[partitionId, kind] : partitionKinds) {
            shiftedPartitionKinds[partitionId >= firstPartitionToShift ? partitionId + 1 : partitionId] = kind;
        }
        partitionKinds = std::move(shiftedPartitionKinds);
    }

    SmallVector<Operation *> ops;
    DenseMap<Operation *, int64_t> partitionByOp;
    DenseMap<int64_t, PartitionKind> partitionKinds;
    DenseSet<Operation *> leafOps;
    DenseMap<Value, int64_t> sequenceOutputIndexByValue;
};

class ModelPartitionMarkingPass : public impl::ModelPartitionMarkingPassBase<ModelPartitionMarkingPass> {
  public:
    void runOnOperation() override {
        mlir::ModuleOp moduleOp = getOperation();

        WalkResult result = moduleOp.walk([&](func::FuncOp funcOp) -> WalkResult {
            if (funcOp.isDeclaration()) {
                funcOp.emitError("model partition marking requires function definitions");
                return WalkResult::interrupt();
            }
            PartitionPlan plan;
            plan.plan(funcOp);
            return WalkResult::advance();
        });

        if (result.wasInterrupted()) {
            signalPassFailure();
        }
    }
};

} // namespace

} // namespace mlir::model_converter_passes
