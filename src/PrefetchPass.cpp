#include "PrefetchAnalysis.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

using namespace llvm;

namespace prefetching {

static std::string valueToString(Value *V) {
    if (!V) return "<null>";
    if (V->hasName()) return std::string(V->getName());

    std::string S;
    raw_string_ostream OS(S);
    V->printAsOperand(OS, false);
    return OS.str();
}

static std::string instToString(Instruction *I) {
    if (!I) return "<null>";

    std::string S;
    raw_string_ostream OS(S);
    I->print(OS);
    return OS.str();
}

static std::string scevToString(const SCEV *Expr) {
    if (!Expr) return "<null>";

    std::string S;
    raw_string_ostream OS(S);
    Expr->print(OS);
    return OS.str();
}

static bool isInterestingLoop(Loop *L) {
    if (!L) return false;
    return L->getLoopDepth() >= 1;
}

/// A very simple heuristic:
/// - base distance = 2
/// - if loop is deeper, increase slightly
/// - clamp to [1, 8]
static int64_t choosePrefetchDistance(Loop *L) {
    if (!L) return 0;
    int64_t D = 2 + static_cast<int64_t>(L->getLoopDepth()) - 1;
    if (D < 1) D = 1;
    if (D > 8) D = 8;
    return D;
}

/// Returns true if Expr mentions the loop's induction recurrence somewhere.
/// This is a simple recursive check over the SCEV tree.
static bool scevMentionsLoop(const SCEV *Expr, Loop *L) {
    if (!Expr || !L) return false;

    if (auto *AR = dyn_cast<SCEVAddRecExpr>(Expr)) {
        if (AR->getLoop() == L)
            return true;
    }

    for (const SCEV *Op : Expr->operands()) {
        if (scevMentionsLoop(Op, L))
            return true;
    }
    return false;
}

/// Very coarse notion of "affine enough for first milestone".
/// Accept constants, unknowns, add recurrences, adds, muls, casts/sext/zext, etc.
/// Reject div/rem and weird cases indirectly by not matching them well later.
static bool isAffineLikeSCEV(const SCEV *Expr) {
    if (!Expr) return false;

    if (isa<SCEVConstant>(Expr) || isa<SCEVUnknown>(Expr))
        return true;

    if (auto *AR = dyn_cast<SCEVAddRecExpr>(Expr)) {
        // Require affine recurrence: start + step*i
        return AR->isAffine();
    }

    if (isa<SCEVAddExpr>(Expr) || isa<SCEVMulExpr>(Expr) ||
        isa<SCEVZeroExtendExpr>(Expr) || isa<SCEVSignExtendExpr>(Expr) ||
        isa<SCEVTruncateExpr>(Expr))
        return true;

    return false;
}

static void analyzeLoopRecursive(Loop *L,
                                 ScalarEvolution &SE,
                                 SmallVectorImpl<PrefetchCandidateInfo> &Out) {
    if (!L) return;

    if (isInterestingLoop(L)) {
        for (BasicBlock *BB : L->blocks()) {
            // Skip blocks belonging to subloops so we don't double count loads.
            if (Loop *SubL = L->getSubLoopContaining(BB)) {
                if (SubL != L)
                    continue;
            }

            for (Instruction &I : *BB) {
                auto *LI = dyn_cast<LoadInst>(&I);
                if (!LI)
                    continue;

                Value *Ptr = LI->getPointerOperand();
                if (!Ptr)
                    continue;

                const SCEV *PtrSCEV = SE.getSCEV(Ptr);

                PrefetchCandidateInfo Info;
                Info.loadText = instToString(LI);
                Info.pointerText = valueToString(Ptr);
                Info.scevText = scevToString(PtrSCEV);
                Info.loopDepth = L->getLoopDepth();
                Info.affine = isAffineLikeSCEV(PtrSCEV);
                Info.loopVariant = scevMentionsLoop(PtrSCEV, L);

                // First milestone rule:
                // candidate only if pointer is affine-like and changes with loop iters.
                if (Info.affine && Info.loopVariant) {
                    Info.prefetchDistance = choosePrefetchDistance(L);
                } else {
                    Info.prefetchDistance = 0;
                }

                Out.push_back(Info);
            }
        }
    }

    for (Loop *SubL : L->getSubLoops())
        analyzeLoopRecursive(SubL, SE, Out);
}

struct PrefetchPass : public PassInfoMixin<PrefetchPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        auto &LI = AM.getResult<LoopAnalysis>(F);
        auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);

        SmallVector<PrefetchCandidateInfo, 16> Candidates;

        for (Loop *TopLevelLoop : LI)
            analyzeLoopRecursive(TopLevelLoop, SE, Candidates);

        if (Candidates.empty())
            return PreservedAnalyses::all();

        errs() << "[PrefetchPass] function=" << F.getName() << "\n";
        for (const auto &C : Candidates) {
            errs() << "  load=" << C.loadText << "\n";
            errs() << "    ptr=" << C.pointerText << "\n";
            errs() << "    scev=" << C.scevText << "\n";
            errs() << "    loop_depth=" << C.loopDepth << "\n";
            errs() << "    affine=" << (C.affine ? "yes" : "no") << "\n";
            errs() << "    loop_variant=" << (C.loopVariant ? "yes" : "no") << "\n";

            if (C.prefetchDistance > 0) {
                errs() << "    candidate=yes prefetch_distance="
                       << C.prefetchDistance << "\n";
            } else {
                errs() << "    candidate=no\n";
            }
        }

        return PreservedAnalyses::all();
    }
};

} // namespace prefetching

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION,
        "PrefetchPass",
        "0.1",
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name,
                   FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "prefetch-pass") {
                        FPM.addPass(prefetching::PrefetchPass());
                        return true;
                    }
                    return false;
                });
        }
    };
}