#include "PrefetchAnalysis.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <set>
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
    return L && L->getLoopDepth() >= 1;
}

// Assumed memory latency (DRAM) and issue width used to estimate how many
// iterations of this loop fit inside one cache-miss latency window.
static constexpr int64_t MEM_LATENCY_CYCLES = 200;
static constexpr int64_t ASSUMED_IPC        = 4;

static int64_t choosePrefetchDistance(Loop *L, LoopInfo &LI) {
    if (!L) return 0;

    // Count instructions only in blocks owned directly by this loop,
    // not in nested subloop blocks (those have their own prefetch decisions).
    int64_t InstCount = 0;
    for (BasicBlock *BB : L->blocks()) {
        if (LI.getLoopFor(BB) != L)
            continue;
        InstCount += static_cast<int64_t>(BB->size());
    }
    if (InstCount == 0)
        InstCount = 1;

    // iteration_cycles ≈ InstCount / IPC
    // distance = ceil(MEM_LATENCY / iteration_cycles)
    //          = ceil(MEM_LATENCY * IPC / InstCount)
    int64_t D = (MEM_LATENCY_CYCLES * ASSUMED_IPC + InstCount - 1) / InstCount;
    if (D < 1) D = 1;
    if (D > 8) D = 8;
    return D;
}

static bool scevMentionsLoop(const SCEV *Expr, Loop *L) {
    if (!Expr || !L) return false;

    if (auto *AR = dyn_cast<SCEVAddRecExpr>(Expr))
        return AR->getLoop() == L;

    if (auto *Add = dyn_cast<SCEVAddExpr>(Expr)) {
        for (const SCEV *Op : Add->operands())
            if (scevMentionsLoop(Op, L))
                return true;
        return false;
    }

    if (auto *Mul = dyn_cast<SCEVMulExpr>(Expr)) {
        for (const SCEV *Op : Mul->operands())
            if (scevMentionsLoop(Op, L))
                return true;
        return false;
    }

    if (auto *Min = dyn_cast<SCEVSMinExpr>(Expr)) {
        for (const SCEV *Op : Min->operands())
            if (scevMentionsLoop(Op, L))
                return true;
        return false;
    }

    if (auto *UMin = dyn_cast<SCEVUMinExpr>(Expr)) {
        for (const SCEV *Op : UMin->operands())
            if (scevMentionsLoop(Op, L))
                return true;
        return false;
    }

    if (auto *Max = dyn_cast<SCEVSMaxExpr>(Expr)) {
        for (const SCEV *Op : Max->operands())
            if (scevMentionsLoop(Op, L))
                return true;
        return false;
    }

    if (auto *UMax = dyn_cast<SCEVUMaxExpr>(Expr)) {
        for (const SCEV *Op : UMax->operands())
            if (scevMentionsLoop(Op, L))
                return true;
        return false;
    }

    if (auto *ZExt = dyn_cast<SCEVZeroExtendExpr>(Expr))
        return scevMentionsLoop(ZExt->getOperand(), L);

    if (auto *SExt = dyn_cast<SCEVSignExtendExpr>(Expr))
        return scevMentionsLoop(SExt->getOperand(), L);

    if (auto *Trunc = dyn_cast<SCEVTruncateExpr>(Expr))
        return scevMentionsLoop(Trunc->getOperand(), L);

    return false;
}

static bool isAffineLikeSCEV(const SCEV *Expr) {
    if (!Expr) return false;

    if (isa<SCEVConstant>(Expr) || isa<SCEVUnknown>(Expr))
        return true;

    if (auto *AR = dyn_cast<SCEVAddRecExpr>(Expr))
        return AR->isAffine();

    if (isa<SCEVAddExpr>(Expr) || isa<SCEVMulExpr>(Expr) ||
        isa<SCEVZeroExtendExpr>(Expr) || isa<SCEVSignExtendExpr>(Expr) ||
        isa<SCEVTruncateExpr>(Expr))
        return true;

    return false;
}

// Strip GEPs, bitcasts, addrspacecasts, etc. back to the underlying base object.
static Value *getUnderlyingBasePointer(Value *V) {
    while (true) {
        if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
            V = GEP->getPointerOperand();
            continue;
        }
        if (auto *BC = dyn_cast<BitCastInst>(V)) {
            V = BC->getOperand(0);
            continue;
        }
        if (auto *ASC = dyn_cast<AddrSpaceCastInst>(V)) {
            V = ASC->getOperand(0);
            continue;
        }
        if (auto *GEP = dyn_cast<GEPOperator>(V)) {
            V = GEP->getPointerOperand();
            continue;
        }
        if (auto *CE = dyn_cast<ConstantExpr>(V)) {
            if (CE->getOpcode() == Instruction::BitCast ||
                CE->getOpcode() == Instruction::AddrSpaceCast ||
                CE->getOpcode() == Instruction::GetElementPtr) {
                V = CE->getOperand(0);
                continue;
            }
        }
        break;
    }
    return V;
}

static std::string basePointerName(Value *Ptr) {
    Value *Base = getUnderlyingBasePointer(Ptr);
    if (auto *A = dyn_cast<Argument>(Base)) {
        // Prefer the IR name if present (optimized builds sometimes keep it).
        if (A->hasName())
            return std::string(A->getName());

        unsigned ArgNo = A->getArgNo();
        Function *F = A->getParent();

        // kernel_arg_name is emitted by some OpenCL toolchains and carries the
        // original source parameter name (e.g. "A", "B").
        if (MDNode *Names = F->getMetadata("kernel_arg_name")) {
            if (ArgNo < Names->getNumOperands()) {
                if (auto *MDS = dyn_cast<MDString>(Names->getOperand(ArgNo))) {
                    StringRef S = MDS->getString();
                    if (!S.empty())
                        return S.str();
                }
            }
        }

        // kernel_arg_type is always present; use it to produce "float*_0" etc.
        if (MDNode *Types = F->getMetadata("kernel_arg_type")) {
            if (ArgNo < Types->getNumOperands()) {
                if (auto *MDS = dyn_cast<MDString>(Types->getOperand(ArgNo))) {
                    StringRef S = MDS->getString();
                    if (!S.empty())
                        return S.str() + "_" + std::to_string(ArgNo);
                }
            }
        }

        return std::string("arg") + std::to_string(ArgNo);
    }
    return "";
}

// Extract the loop's upper bound value (e.g. N in "for (i = ...; i < N; ...)").
// Works only when there is a single exiting block with a simple icmp condition.
static Value *findLoopBound(Loop *L) {
    BasicBlock *ExitingBB = L->getExitingBlock(); // null if multiple exit edges
    if (!ExitingBB) return nullptr;

    auto *BI = dyn_cast<BranchInst>(ExitingBB->getTerminator());
    if (!BI || !BI->isConditional()) return nullptr;

    auto *Cmp = dyn_cast<ICmpInst>(BI->getCondition());
    if (!Cmp) return nullptr;

    // "i < N" / "i <= N" / unsigned variants — bound is the RHS operand.
    ICmpInst::Predicate P = Cmp->getPredicate();
    if (P == ICmpInst::ICMP_SLT || P == ICmpInst::ICMP_SLE ||
        P == ICmpInst::ICMP_ULT || P == ICmpInst::ICMP_ULE)
        return Cmp->getOperand(1);

    return nullptr;
}

static void analyzeLoopRecursive(Loop *L,
                                 LoopInfo &LI,
                                 ScalarEvolution &SE,
                                 SmallVectorImpl<PrefetchCandidateInfo> &Out) {
    if (!L) return;

    if (isInterestingLoop(L)) {
        for (BasicBlock *BB : L->blocks()) {
            if (LI.getLoopFor(BB) != L)
                continue;

            for (Instruction &I : *BB) {
                auto *LoadI = dyn_cast<LoadInst>(&I);
                if (!LoadI)
                    continue;

                Value *Ptr = LoadI->getPointerOperand();
                if (!Ptr)
                    continue;

                // Only keep loads derived from kernel pointer arguments like A/B.
                std::string BaseName = basePointerName(Ptr);
                if (BaseName.empty())
                    continue;

                const SCEV *PtrSCEV = SE.getSCEV(Ptr);

                PrefetchCandidateInfo Info;
                Info.loadInst = LoadI;
                Info.loop     = L;
                Info.loadText = instToString(LoadI);
                Info.pointerText = valueToString(Ptr);
                Info.baseText  = BaseName;   // normalized name, e.g. "A"
                Info.scevText = scevToString(PtrSCEV);
                Info.loopDepth = L->getLoopDepth();
                Info.affine = isAffineLikeSCEV(PtrSCEV);
                Info.loopVariant = scevMentionsLoop(PtrSCEV, L);

                if (Info.affine && Info.loopVariant) {
                    Info.prefetchDistance = choosePrefetchDistance(L, LI);

                    // Bounds check: use static trip count to decide if the prefetch
                    // at (i + prefetchDistance) is guaranteed to stay in bounds.
                    unsigned TripCount = SE.getSmallConstantTripCount(L);
                    if (TripCount > 0) {
                        Info.boundsStatus = (static_cast<int64_t>(TripCount) > Info.prefetchDistance)
                            ? BoundsStatus::Safe
                            : BoundsStatus::Unsafe;
                    } else {
                        Info.boundsStatus = BoundsStatus::NeedsRuntimeGuard;
                    }

                    // Stride extraction: peel extension/truncation wrappers, then
                    // pull the step from the innermost affine AddRec.
                    const SCEV *Inner = PtrSCEV;
                    while (true) {
                        if (auto *Z = dyn_cast<SCEVZeroExtendExpr>(Inner)) { Inner = Z->getOperand(); continue; }
                        if (auto *S = dyn_cast<SCEVSignExtendExpr>(Inner)) { Inner = S->getOperand(); continue; }
                        if (auto *T = dyn_cast<SCEVTruncateExpr>(Inner))   { Inner = T->getOperand(); continue; }
                        break;
                    }
                    if (auto *AR = dyn_cast<SCEVAddRecExpr>(Inner)) {
                        if (AR->isAffine()) {
                            if (auto *SC = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE)))
                                Info.strideBytes = SC->getValue()->getSExtValue();
                        }
                    }

                    // Classify benefit based on stride vs. cache line size (64 B).
                    // A prefetch warms one cache line; smaller strides mean one
                    // prefetch covers more iterations.
                    int64_t AbsStride = Info.strideBytes < 0
                                        ? -Info.strideBytes : Info.strideBytes;
                    if (AbsStride == 0)
                        Info.benefit = PrefetchBenefit::Unknown;
                    else if (AbsStride <= 64)
                        Info.benefit = PrefetchBenefit::High;
                    else if (AbsStride <= 256)
                        Info.benefit = PrefetchBenefit::Medium;
                    else
                        Info.benefit = PrefetchBenefit::Low;
                } else {
                    Info.prefetchDistance = 0;
                    Info.boundsStatus = BoundsStatus::NotApplicable;
                }

                Out.push_back(Info);
            }
        }
    }

    for (Loop *SubL : L->getSubLoops())
        analyzeLoopRecursive(SubL, LI, SE, Out);
}

struct PrefetchPass : public PassInfoMixin<PrefetchPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        auto &LI = AM.getResult<LoopAnalysis>(F);
        auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);

        SmallVector<PrefetchCandidateInfo, 16> Candidates;
        for (Loop *TopLevelLoop : LI)
            analyzeLoopRecursive(TopLevelLoop, LI, SE, Candidates);
        // redundant compute
        if (Candidates.empty()) {
            errs() << "[PrefetchPass] function=" << F.getName()
                << " no prefetch candidates found\n";
            return PreservedAnalyses::all();
        }

        std::set<std::string> Seen;
        errs() << "[PrefetchPass] function=" << F.getName() << "\n";

        for (const auto &C : Candidates) {
            std::string Key = C.baseText + "|depth=" + std::to_string(C.loopDepth)
                            + "|scev=" + C.scevText;
            if (!Seen.insert(Key).second)
                continue;

            errs() << "  access=" << C.pointerText
                << " base=" << C.baseText
                << " affine=" << (C.affine ? "yes" : "no")
                << " loop_variant=" << (C.loopVariant ? "yes" : "no");

            if (C.prefetchDistance > 0) {
                const char *bounds =
                    C.boundsStatus == BoundsStatus::Safe              ? "yes"          :
                    C.boundsStatus == BoundsStatus::Unsafe             ? "no"           :
                                                                         "runtime_guard";
                const char *benefit =
                    C.benefit == PrefetchBenefit::High    ? "high"    :
                    C.benefit == PrefetchBenefit::Medium  ? "medium"  :
                    C.benefit == PrefetchBenefit::Low     ? "low"     :
                                                            "unknown";
                errs() << " candidate=yes"
                       << " prefetch_distance=" << C.prefetchDistance
                       << " bounds_safe=" << bounds
                       << " benefit=" << benefit;
                if (C.strideBytes != 0)
                    errs() << " stride_bytes=" << C.strideBytes;
            } else {
                errs() << " candidate=no";
            }

            errs() << "\n";
        }

        // --- Transformation phase ---
        LLVMContext &Ctx = F.getContext();
        const DataLayout &DL = F.getParent()->getDataLayout();
        Type *I64Ty   = Type::getInt64Ty(Ctx);
        Type *I8PtrTy = Type::getInt8PtrTy(Ctx);
        Function *PrefetchFn = Intrinsic::getDeclaration(
            F.getParent(), Intrinsic::prefetch, {I8PtrTy});
        bool Changed = false;

        for (const auto &C : Candidates) {
            if (C.prefetchDistance == 0) continue;
            if (C.strideBytes == 0)      continue;
            if (!C.loadInst || !C.loop)  continue;
            if (C.boundsStatus == BoundsStatus::Unsafe ||
                C.boundsStatus == BoundsStatus::NotApplicable) continue;

            Type    *ElemTy  = C.loadInst->getType();
            uint64_t ElemSize = DL.getTypeAllocSize(ElemTy);
            if (ElemSize == 0 || C.strideBytes % (int64_t)ElemSize != 0) continue;

            int64_t AheadElems = C.prefetchDistance * (C.strideBytes / (int64_t)ElemSize);
            Value  *Ptr        = C.loadInst->getPointerOperand();

            // Instructions inserted before C.loadInst land in the current block
            // and will remain there after any block split below.
            IRBuilder<> Builder(C.loadInst);

            Value *PrefetchPtr = Builder.CreateGEP(
                ElemTy, Ptr,
                ConstantInt::get(I64Ty, AheadElems), "prefetch.ptr");
            Value *I8Ptr = Builder.CreateBitCast(PrefetchPtr, I8PtrTy, "prefetch.i8ptr");

            if (C.boundsStatus == BoundsStatus::Safe) {
                // Static trip count proves we are in bounds — insert unconditionally.
                Builder.CreateCall(PrefetchFn,
                    {I8Ptr, Builder.getInt32(0), Builder.getInt32(3), Builder.getInt32(1)});
                errs() << "[PrefetchPass] inserted prefetch (safe): base=" << C.baseText
                       << " ahead_elems=" << AheadElems << "\n";
                Changed = true;

            } else { // NeedsRuntimeGuard
                Value *Bound = findLoopBound(C.loop);
                if (!Bound) continue; // can't determine loop bound, skip

                // Check: (prefetch_ptr - base) < N * elem_size, all in bytes.
                // Using pointer arithmetic avoids recomputing the element index.
                Value *PrefetchInt  = Builder.CreatePtrToInt(PrefetchPtr, I64Ty);
                Value *BaseInt      = Builder.CreatePtrToInt(
                    getUnderlyingBasePointer(Ptr), I64Ty);
                Value *ByteOffset   = Builder.CreateSub(PrefetchInt, BaseInt,
                    "prefetch.byte_offset");
                Value *Bound64      = Builder.CreateIntCast(Bound, I64Ty, /*isSigned=*/true);
                Value *BoundBytes   = Builder.CreateMul(
                    Bound64, ConstantInt::get(I64Ty, ElemSize), "prefetch.bound_bytes");
                Value *InBounds     = Builder.CreateICmpULT(
                    ByteOffset, BoundBytes, "prefetch.inbounds");

                // Split the block at C.loadInst; the load moves to MergeBB.
                // GEP/bitcast/cmp instructions above stay in OrigBB.
                BasicBlock *OrigBB  = C.loadInst->getParent();
                BasicBlock *MergeBB = OrigBB->splitBasicBlock(C.loadInst, "prefetch.merge");
                BasicBlock *PrefetchBB = BasicBlock::Create(
                    Ctx, "prefetch.block", OrigBB->getParent(), MergeBB);

                // Replace the unconditional branch splitBasicBlock inserted.
                OrigBB->getTerminator()->eraseFromParent();
                IRBuilder<>(OrigBB).CreateCondBr(InBounds, PrefetchBB, MergeBB);

                // Prefetch block: call the intrinsic then fall through to merge.
                IRBuilder<> PB(PrefetchBB);
                PB.CreateCall(PrefetchFn,
                    {I8Ptr, PB.getInt32(0), PB.getInt32(3), PB.getInt32(1)});
                PB.CreateBr(MergeBB);

                errs() << "[PrefetchPass] inserted prefetch (guarded): base=" << C.baseText
                       << " ahead_elems=" << AheadElems << "\n";
                Changed = true;
            }
        }

        return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
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