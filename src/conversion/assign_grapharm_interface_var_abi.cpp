/*
 * SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#include "include/passes.hpp"

#include "llvm/ADT/STLExtras.h"

namespace mlir::model_converter_passes {
#define GEN_PASS_DEF_ASSIGNGRAPHARMINTERFACEVARABIPASS
#include "passes.hpp.inc"
namespace {

constexpr StringLiteral graphARMInterfaceVarABIAttrName = "spv.grapharm.interface_var_abi";
constexpr uint32_t graphARMDescriptorSet = 0;

class AssignGraphARMInterfaceVarABIPass
    : public impl::AssignGraphARMInterfaceVarABIPassBase<AssignGraphARMInterfaceVarABIPass> {
  public:
    using impl::AssignGraphARMInterfaceVarABIPassBase<
        AssignGraphARMInterfaceVarABIPass>::AssignGraphARMInterfaceVarABIPassBase;

    void runOnOperation() override {
        if (failed(assignInterfaceVarABI(getOperation()))) {
            signalPassFailure();
        }
    }

  private:
    uint32_t getBindingId(const DenseMap<Value, uint32_t> &bindingIds, Value value) {
        auto it = bindingIds.find(value);
        assert(it != bindingIds.end() && "expected binding for graph interface value");
        return it->second;
    }

    spirv::InterfaceVarABIAttr getInterfaceVarABIAttr(uint32_t bindingId) {
        return spirv::InterfaceVarABIAttr::get(graphARMDescriptorSet, bindingId, std::nullopt, &getContext());
    }

    LogicalResult assignInterfaceVarABI(vgf::SequenceOp sequenceOp) {
        auto isSequenceInputOperand = [&](Value operand) {
            return llvm::is_contained(sequenceOp.getArguments(), operand);
        };

        Operation *sequenceOutputOp = sequenceOp.front().getTerminator();
        auto isSequenceOutputOperand = [&](Value operand) {
            return llvm::is_contained(sequenceOutputOp->getOperands(), operand);
        };

        DenseMap<Value, uint32_t> bindingIds;
        auto assignBindingIfUnset = [&](Value operand, uint32_t bindingId) {
            bindingIds.try_emplace(operand, bindingId);
        };

        // First: Resolve sequence inputs.
        for (BlockArgument operand : sequenceOp.getArguments()) {
            sequenceOp.walk([&](vgf::SegmentOp segmentOp) {
                Operation *runSegmentOp = segmentOp->getNextNode();
                if (runSegmentOp && llvm::is_contained(runSegmentOp->getOperands(), operand)) {
                    assignBindingIfUnset(operand, static_cast<uint32_t>(operand.getArgNumber()));
                }
            });
        }

        // Second: Resolve intermediate values.
        auto bindingId = static_cast<uint32_t>(sequenceOp.getNumArguments());
        WalkResult sequenceWalkResult = sequenceOp.walk([&](vgf::SegmentOp segmentOp) {
            Operation *runSegmentOp = segmentOp->getNextNode();
            if (!runSegmentOp) {
                return WalkResult::advance();
            }

            for (Value operand : runSegmentOp->getOperands()) {
                if (!isSequenceInputOperand(operand) && !bindingIds.contains(operand)) {
                    bindingIds[operand] = bindingId++;
                }
            }
            for (Value result : runSegmentOp->getResults()) {
                if (!isSequenceOutputOperand(result) && !bindingIds.contains(result)) {
                    bindingIds[result] = bindingId++;
                }
            }
            return WalkResult::advance();
        });
        if (sequenceWalkResult.wasInterrupted()) {
            return failure();
        }

        // Third: Resolve sequence outputs.
        for (Value operand : sequenceOutputOp->getOperands()) {
            sequenceOp.walk([&](vgf::SegmentOp segmentOp) {
                Operation *runSegmentOp = segmentOp->getNextNode();
                if (runSegmentOp && llvm::is_contained(runSegmentOp->getResults(), operand) &&
                    !bindingIds.contains(operand)) {
                    bindingIds[operand] = bindingId++;
                }
            });
        }

        sequenceWalkResult = sequenceOp.walk([&](vgf::SegmentOp segmentOp) {
            if (segmentOp.getSegmentType() != vgf::SegmentTypeEnum::GRAPH) {
                return WalkResult::advance();
            }

            Operation *runSegmentOp = segmentOp->getNextNode();
            if (!runSegmentOp) {
                segmentOp.emitError("expected graph segment to be followed by a segment run op");
                return WalkResult::interrupt();
            }

            auto funcRange = segmentOp.getOps<func::FuncOp>();
            auto funcIt = funcRange.begin();
            if (funcIt == funcRange.end()) {
                segmentOp.emitError("expected graph segment to contain a func.func");
                return WalkResult::interrupt();
            }
            func::FuncOp funcOp = *funcIt;
            ++funcIt;
            if (funcIt != funcRange.end()) {
                segmentOp.emitError("expected graph segment to contain a single func.func");
                return WalkResult::interrupt();
            }

            if (runSegmentOp->getNumOperands() != funcOp.getNumArguments()) {
                funcOp.emitError("segment run operand count does not match graph function arguments");
                return WalkResult::interrupt();
            }
            if (runSegmentOp->getNumResults() != funcOp.getNumResults()) {
                funcOp.emitError("segment run result count does not match graph function results");
                return WalkResult::interrupt();
            }

            for (auto [argIndex, operand] : llvm::enumerate(runSegmentOp->getOperands())) {
                funcOp.setArgAttr(static_cast<unsigned>(argIndex), graphARMInterfaceVarABIAttrName,
                                  getInterfaceVarABIAttr(getBindingId(bindingIds, operand)));
            }

            for (auto [resultIndex, result] : llvm::enumerate(runSegmentOp->getResults())) {
                funcOp.setResultAttr(static_cast<unsigned>(resultIndex), graphARMInterfaceVarABIAttrName,
                                     getInterfaceVarABIAttr(getBindingId(bindingIds, result)));
            }

            return WalkResult::advance();
        });

        return sequenceWalkResult.wasInterrupted() ? failure() : success();
    }
};

} // namespace
} // namespace mlir::model_converter_passes
