#include "CoalescingAnalysis.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>

using namespace llvm;

namespace coalescing {

static std::string valueToString(Value *V) {
    if (!V) return "<null>";
    if (V->hasName()) return std::string(V->getName());
    std::string S;
    raw_string_ostream OS(S);
    V->printAsOperand(OS, false);
    return OS.str();
}

static bool isConstInt(Value *V, int64_t &Out) {
    if (auto *CI = dyn_cast<ConstantInt>(V)) {
        Out = CI->getSExtValue();
        return true;
    }
    return false;
}

static bool isGlobalIdCall(Value *V) {
    if (auto *CI = dyn_cast<CallInst>(V)) {
        if (Function *Callee = CI->getCalledFunction()) {
            StringRef N = Callee->getName();
            return N.contains("get_global_id") || N.contains("_Z13get_global_idj");
        }
    }
    return false;
}

struct LinearExpr {
    bool threadDependent = false;
    int64_t stride = 0;
    int64_t offset = 0;
    std::string pretty;
};

static std::optional<LinearExpr> parseLinearIndex(Value *V) {
    if (!V) return std::nullopt;

    if (isGlobalIdCall(V)) {
        return LinearExpr{true, 1, 0, "tid"};
    }

    int64_t C = 0;
    if (isConstInt(V, C)) {
        return LinearExpr{false, 0, C, std::to_string(C)};
    }

    if (auto *BO = dyn_cast<BinaryOperator>(V)) {
        auto L = parseLinearIndex(BO->getOperand(0));
        auto R = parseLinearIndex(BO->getOperand(1));
        if (!L || !R) return std::nullopt;

        switch (BO->getOpcode()) {
            case Instruction::Add: {
                if (L->threadDependent && !R->threadDependent) {
                    return LinearExpr{true, L->stride, L->offset + R->offset, L->pretty + "+" + R->pretty};
                }
                if (!L->threadDependent && R->threadDependent) {
                    return LinearExpr{true, R->stride, R->offset + L->offset, L->pretty + "+" + R->pretty};
                }
                if (!L->threadDependent && !R->threadDependent) {
                    return LinearExpr{false, 0, L->offset + R->offset, L->pretty + "+" + R->pretty};
                }
                return std::nullopt;
            }
            case Instruction::Mul: {
                if (L->threadDependent && !R->threadDependent) {
                    return LinearExpr{true, L->stride * R->offset, L->offset * R->offset, L->pretty + "*" + R->pretty};
                }
                if (!L->threadDependent && R->threadDependent) {
                    return LinearExpr{true, R->stride * L->offset, R->offset * L->offset, L->pretty + "*" + R->pretty};
                }
                if (!L->threadDependent && !R->threadDependent) {
                    return LinearExpr{false, 0, L->offset * R->offset, L->pretty + "*" + R->pretty};
                }
                return std::nullopt;
            }
            case Instruction::Sub: {
                if (L->threadDependent && !R->threadDependent) {
                    return LinearExpr{true, L->stride, L->offset - R->offset, L->pretty + "-" + R->pretty};
                }
                if (!L->threadDependent && !R->threadDependent) {
                    return LinearExpr{false, 0, L->offset - R->offset, L->pretty + "-" + R->pretty};
                }
                return std::nullopt;
            }
            default:
                return std::nullopt;
        }
    }

    if (auto *Cast = dyn_cast<CastInst>(V)) {
        return parseLinearIndex(Cast->getOperand(0));
    }

    return std::nullopt;
}

int64_t estimateTransactionsPerWarp(int64_t stride, int64_t elementSizeBytes, int64_t warpSize, int64_t segmentSizeBytes) {
    if (stride <= 0 || elementSizeBytes <= 0) return 0;
    int64_t spanBytes = (((warpSize - 1) * stride) + 1) * elementSizeBytes;
    return (spanBytes + segmentSizeBytes - 1) / segmentSizeBytes;
}

std::string classifyAccess(bool threadDependent, int64_t stride, int64_t estimatedTransactions) {
    if (!threadDependent) return "UNKNOWN";
    if (stride == 1 && estimatedTransactions <= 1) return "COALESCED";
    if (stride == 1) return "LIKELY_COALESCED";
    if (stride > 1) return "NON_COALESCED";
    return "UNKNOWN";
}

static int64_t getElementSizeBytes(GetElementPtrInst *GEP) {
    Type *Ty = GEP->getResultElementType();
    if (!Ty) return 0;
    if (Ty->isFloatTy()) return 4;
    if (Ty->isDoubleTy()) return 8;
    if (Ty->isIntegerTy()) return Ty->getIntegerBitWidth() / 8;
    return 4;
}

struct CoalescingPass : public PassInfoMixin<CoalescingPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
        bool PrintedHeader = false;
        for (Instruction &I : instructions(F)) {
            auto *GEP = dyn_cast<GetElementPtrInst>(&I);
            if (!GEP || GEP->getNumOperands() < 2) continue;

            Value *Index = GEP->getOperand(GEP->getNumOperands() - 1);
            auto Parsed = parseLinearIndex(Index);
            int64_t ElemSize = getElementSizeBytes(GEP);

            if (!PrintedHeader) {
                errs() << "[CoalescingPass] function=" << F.getName() << "\n";
                PrintedHeader = true;
            }

            if (!Parsed) {
                errs() << "  access=" << valueToString(GEP)
                       << " thread_dependent=unknown stride=unknown estimated_transactions=unknown class=UNKNOWN\n";
                continue;
            }

            int64_t Tx = estimateTransactionsPerWarp(std::max<int64_t>(1, Parsed->stride), ElemSize);
            std::string C = classifyAccess(Parsed->threadDependent, Parsed->stride, Tx);

            errs() << "  access=" << Parsed->pretty
                   << " thread_dependent=" << (Parsed->threadDependent ? "yes" : "no")
                   << " stride=" << Parsed->stride
                   << " estimated_transactions=" << Tx
                   << " class=" << C << "\n";
        }
        return PreservedAnalyses::all();
    }
};

} // namespace coalescing

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION,
        "CoalescingPass",
        "0.1",
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "coalescing-pass") {
                        FPM.addPass(coalescing::CoalescingPass());
                        return true;
                    }
                    return false;
                });
        }
    };
}
