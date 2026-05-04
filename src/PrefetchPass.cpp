#include "PrefetchAnalysis.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Operator.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
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

// A100 L2→HBM latency (~480 cycles at 1.41 GHz). On GPU, effective IPC per
// warp is ~1 (warp scheduler, not superscalar), so distance ≈ latency / InstCount.
static constexpr int64_t MEM_LATENCY_CYCLES = 480;
static constexpr int64_t ASSUMED_IPC        = 1;

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
    if (D < 1)  D = 1;
    if (D > 16) D = 16;  // A100: deeper HBM latency warrants longer lookahead
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
                            const SCEV *Step = AR->getStepRecurrence(SE);
                            if (auto *SC = dyn_cast<SCEVConstant>(Step))
                                Info.strideBytes = SC->getValue()->getSExtValue();
                            else
                                Info.strideSCEV = Step; // runtime stride (e.g. B[k*nj+j])
                        }
                    }

                    // Classify benefit. Symbolic stride (e.g. column-stride B[k*nj+j])
                    // gets Unknown since we can't know the value statically, but it is
                    // still a valid prefetch candidate and will be transformed.
                    int64_t AbsStride = Info.strideBytes < 0
                                        ? -Info.strideBytes : Info.strideBytes;
                    if (Info.strideSCEV)
                        Info.benefit = PrefetchBenefit::Unknown;
                    else if (AbsStride == 0)
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

enum class PrefetchMode { Hint, Pipeline };

// Software-pipeline a candidate load: move its data one iteration earlier by
// loading into a register in the previous iteration, then using that register
// value in the current one.  This is the paper's §3.6 register-prefetch pattern.
//
// Transformation sketch (single-block loop):
//
//   preheader:  prolog_val = load(ptr @ iter 0)         ; NEW
//   header:     val.cur = phi [prolog_val, ph], [val.next, next_load_bb]  ; NEW
//               ... existing PHIs updated: Latch → NextLoadBB ...
//   body:       ... val.cur used in place of original load ...
//   latch:      cond = (i+1 < N)
//               br cond, next_load_bb, exit             ; true edge redirected
//
//   next_load_bb:                                        ; NEW
//               next_ptr = gep(LoadPtr, stride_elems)
//               val.next = load(next_ptr)
//               br header
//
// Restrictions: constant stride only, single preheader + latch.
static bool tryInsertSoftwarePipeline(const PrefetchCandidateInfo &C,
                                       ScalarEvolution &SE,
                                       const DataLayout &DL,
                                       SCEVExpander &Expander) {
    if (!C.loadInst || !C.loop) return false;
    if (C.strideBytes == 0 || C.strideSCEV) return false; // constant stride only

    Loop *L             = C.loop;
    BasicBlock *Preheader = L->getLoopPreheader();
    BasicBlock *Header    = L->getHeader();
    BasicBlock *Latch     = L->getLoopLatch();
    if (!Preheader || !Latch || !Header) return false;

    auto *LatchBI = dyn_cast<BranchInst>(Latch->getTerminator());
    if (!LatchBI || !LatchBI->isConditional()) return false;

    // Which successor of Latch is the back-edge to Header?
    int HeaderSuccIdx = -1;
    for (unsigned i = 0; i < LatchBI->getNumSuccessors(); ++i)
        if (LatchBI->getSuccessor(i) == Header) { HeaderSuccIdx = (int)i; break; }
    if (HeaderSuccIdx == -1) return false;

    LoadInst *LI      = C.loadInst;
    Type     *ElemTy  = LI->getType();
    uint64_t  ElemSize = DL.getTypeAllocSize(ElemTy);
    if (ElemSize == 0) return false;
    if (C.strideBytes % (int64_t)ElemSize != 0) return false;
    int64_t StrideElems = C.strideBytes / (int64_t)ElemSize;

    LLVMContext &Ctx  = LI->getContext();
    Type        *I64Ty = Type::getInt64Ty(Ctx);
    Value *LoadPtr    = LI->getPointerOperand();

    // --- All preconditions satisfied.  Begin IR modifications. ---

    // Step 1: Prolog load in the preheader (value for iteration 0).
    //
    // Expanding the full PtrSCEV ({start,+,stride}) can cause SCEVExpander to
    // introduce a new loop counter PHI and its LCSSA form in the exit block,
    // then reference the LCSSA value in the preheader — a forward-use that
    // breaks SSA dominance.  Instead we expand only AR->getStart(), which is
    // by construction loop-invariant: the expander will never insert loop PHIs
    // for it and all referenced values are available before the loop.
    const SCEV *PtrSCEV = SE.getSCEV(LoadPtr);

    // Peel extensions to reach the affine AddRec (same as analysis phase).
    const SCEV *Inner = PtrSCEV;
    while (true) {
        if (auto *Z = dyn_cast<SCEVZeroExtendExpr>(Inner)) { Inner = Z->getOperand(); continue; }
        if (auto *S = dyn_cast<SCEVSignExtendExpr>(Inner)) { Inner = S->getOperand(); continue; }
        if (auto *T = dyn_cast<SCEVTruncateExpr>(Inner))   { Inner = T->getOperand(); continue; }
        break;
    }
    auto *AR = dyn_cast<SCEVAddRecExpr>(Inner);
    if (!AR || !AR->isAffine() || AR->getLoop() != L) return false;

    const SCEV *StartSCEV = AR->getStart(); // loop-invariant by definition
    Type  *SCEVTy         = StartSCEV->getType();
    Instruction *PreTerm  = Preheader->getTerminator();

    Value *PrologPtrVal;
    if (SCEVTy->isPointerTy()) {
        PrologPtrVal = Expander.expandCodeFor(StartSCEV, SCEVTy, PreTerm);
    } else {
        // SCEV is an integer byte address — expand then inttoptr.
        Value *AddrInt = Expander.expandCodeFor(StartSCEV, SCEVTy, PreTerm);
        IRBuilder<> B(PreTerm);
        PrologPtrVal = B.CreateIntToPtr(AddrInt, LoadPtr->getType(), "pipeline.prolog_ptr");
    }

    IRBuilder<> PreBuilder(PreTerm);
    LoadInst *PrologLoad = PreBuilder.CreateLoad(ElemTy, PrologPtrVal, "pipeline.prolog");
    PrologLoad->setAlignment(LI->getAlign());

    // Step 2: Create NextLoadBB and redirect the latch's back-edge through it.
    Function   *F         = Header->getParent();
    BasicBlock *NextLoadBB = BasicBlock::Create(Ctx, "pipeline.next_load", F, Header);
    LatchBI->setSuccessor(HeaderSuccIdx, NextLoadBB);

    // Step 3: In NextLoadBB, load the next iteration's value and branch to header.
    IRBuilder<> NLB(NextLoadBB);
    Value *NextPtr = NLB.CreateGEP(ElemTy, LoadPtr,
                                    ConstantInt::get(I64Ty, StrideElems),
                                    "pipeline.next_ptr");
    LoadInst *NextLoad = NLB.CreateLoad(ElemTy, NextPtr, "pipeline.next_val");
    NextLoad->setAlignment(LI->getAlign());
    NLB.CreateBr(Header);

    // Step 4: Fix existing header PHIs — their Latch incoming is now from NextLoadBB.
    for (PHINode &PN : Header->phis())
        PN.replaceIncomingBlockWith(Latch, NextLoadBB);

    // Step 5: Insert new PHI carrying the pre-loaded value.
    PHINode *ValPHI = PHINode::Create(ElemTy, 2, "pipeline.cur");
    ValPHI->insertBefore(*Header, Header->getFirstNonPHIIt());
    ValPHI->addIncoming(PrologLoad, Preheader);
    ValPHI->addIncoming(NextLoad, NextLoadBB);

    // Step 6: Replace original load uses and erase it.
    LI->replaceAllUsesWith(ValPHI);
    LI->eraseFromParent();

    errs() << "[PrefetchPass] software-pipeline inserted: base=" << C.baseText
           << " stride_elems=" << StrideElems << "\n";
    return true;
}

struct PrefetchPass : public PassInfoMixin<PrefetchPass> {
    PrefetchMode Mode;
    explicit PrefetchPass(PrefetchMode M = PrefetchMode::Hint) : Mode(M) {}

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
                else if (C.strideSCEV)
                    errs() << " stride=symbolic";
            } else {
                errs() << " candidate=no";
            }

            errs() << "\n";
        }

        // --- Transformation phase ---
        LLVMContext &Ctx = F.getContext();
        const DataLayout &DL = F.getParent()->getDataLayout();
        Type *I64Ty = Type::getInt64Ty(Ctx);
        bool Changed = false;

        SCEVExpander Expander(SE, DL, "prefetch");

        if (Mode == PrefetchMode::Pipeline) {
            for (const auto &C : Candidates) {
                if (C.prefetchDistance == 0) continue;
                if (!C.affine || !C.loopVariant)  continue;
                Changed |= tryInsertSoftwarePipeline(C, SE, DL, Expander);
            }
            return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
        }

        // Hint mode: PTX inline asm emits a real L2 prefetch instruction.
        // llvm.prefetch is silently dropped by the NVPTX backend; this is not.
        // Constraint "l" = 64-bit integer register; [$0+0] = address operand.
        FunctionType *AsmFTy = FunctionType::get(Type::getVoidTy(Ctx), {I64Ty}, false);
        InlineAsm *PrefetchAsm = InlineAsm::get(
            AsmFTy, "prefetch.global.L2 [$0+0];", "l", /*hasSideEffects=*/true);

        for (const auto &C : Candidates) {
            if (C.prefetchDistance == 0) continue;
            if (C.strideBytes == 0 && !C.strideSCEV) continue;
            if (!C.loadInst || !C.loop)  continue;
            if (C.boundsStatus == BoundsStatus::Unsafe ||
                C.boundsStatus == BoundsStatus::NotApplicable) continue;

            Type    *ElemTy  = C.loadInst->getType();
            uint64_t ElemSize = DL.getTypeAllocSize(ElemTy);
            if (ElemSize == 0) continue;

            Value *Ptr = C.loadInst->getPointerOperand();
            IRBuilder<> Builder(C.loadInst);

            // Compute the prefetch address and (for bounds check) the step in bytes.
            Value *PrefetchAddr  = nullptr;
            Value *StepBytesVal  = nullptr; // non-null for symbolic stride

            if (C.strideBytes != 0) {
                // Constant stride: simple element-count GEP.
                if (C.strideBytes % (int64_t)ElemSize != 0) continue;
                int64_t AheadElems = C.prefetchDistance * (C.strideBytes / (int64_t)ElemSize);
                Value *PrefetchPtr = Builder.CreateGEP(
                    ElemTy, Ptr, ConstantInt::get(I64Ty, AheadElems), "prefetch.ptr");
                PrefetchAddr = Builder.CreatePtrToInt(PrefetchPtr, I64Ty, "prefetch.addr");
            } else {
                // Symbolic stride (e.g. B[k*nj+j] where nj is a runtime parameter).
                // Expand the SCEV step to a Value at the loop preheader so it is
                // computed once per kernel invocation, not once per loop iteration.
                BasicBlock *Preheader = C.loop->getLoopPreheader();
                if (!Preheader) continue;

                StepBytesVal = Expander.expandCodeFor(
                    C.strideSCEV, I64Ty, Preheader->getTerminator());

                Value *AheadBytes = Builder.CreateMul(
                    StepBytesVal, ConstantInt::get(I64Ty, C.prefetchDistance),
                    "prefetch.ahead_bytes");
                Value *PtrInt = Builder.CreatePtrToInt(Ptr, I64Ty, "prefetch.ptr_int");
                PrefetchAddr = Builder.CreateAdd(PtrInt, AheadBytes, "prefetch.addr");
            }

            if (C.boundsStatus == BoundsStatus::Safe) {
                Builder.CreateCall(PrefetchAsm, {PrefetchAddr});
                errs() << "[PrefetchPass] inserted prefetch (safe): base=" << C.baseText
                       << " distance=" << C.prefetchDistance
                       << (C.strideSCEV ? " stride=symbolic" : "") << "\n";
                Changed = true;

            } else { // NeedsRuntimeGuard
                Value *Bound = findLoopBound(C.loop);
                if (!Bound) continue;

                Value *BaseAddr   = Builder.CreatePtrToInt(
                    getUnderlyingBasePointer(Ptr), I64Ty, "prefetch.base");
                Value *ByteOffset = Builder.CreateSub(PrefetchAddr, BaseAddr,
                    "prefetch.byte_offset");
                Value *Bound64    = Builder.CreateIntCast(Bound, I64Ty, /*isSigned=*/true);

                // For symbolic stride the total array span is loop_bound * |step_bytes|,
                // which is a tighter bound than loop_bound * elemSize.
                Value *BoundBytes;
                if (StepBytesVal) {
                    Value *AbsStep = Builder.CreateSelect(
                        Builder.CreateICmpSGE(StepBytesVal, ConstantInt::get(I64Ty, 0)),
                        StepBytesVal,
                        Builder.CreateNeg(StepBytesVal, "step.neg"),
                        "step.abs");
                    BoundBytes = Builder.CreateMul(Bound64, AbsStep, "prefetch.bound_bytes");
                } else {
                    BoundBytes = Builder.CreateMul(
                        Bound64, ConstantInt::get(I64Ty, ElemSize), "prefetch.bound_bytes");
                }

                Value *InBounds = Builder.CreateICmpULT(
                    ByteOffset, BoundBytes, "prefetch.inbounds");

                // Split at the load; all address computation stays in OrigBB.
                BasicBlock *OrigBB    = C.loadInst->getParent();
                BasicBlock *MergeBB   = OrigBB->splitBasicBlock(C.loadInst, "prefetch.merge");
                BasicBlock *PrefetchBB = BasicBlock::Create(
                    Ctx, "prefetch.block", OrigBB->getParent(), MergeBB);

                OrigBB->getTerminator()->eraseFromParent();
                IRBuilder<>(OrigBB).CreateCondBr(InBounds, PrefetchBB, MergeBB);

                IRBuilder<> PB(PrefetchBB);
                PB.CreateCall(PrefetchAsm, {PrefetchAddr});
                PB.CreateBr(MergeBB);

                errs() << "[PrefetchPass] inserted prefetch (guarded): base=" << C.baseText
                       << " distance=" << C.prefetchDistance
                       << (C.strideSCEV ? " stride=symbolic" : "") << "\n";
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
                    if (Name == "prefetch-pass" || Name == "prefetch-hint-pass") {
                        FPM.addPass(prefetching::PrefetchPass(prefetching::PrefetchMode::Hint));
                        return true;
                    }
                    if (Name == "prefetch-pipeline-pass") {
                        FPM.addPass(prefetching::PrefetchPass(prefetching::PrefetchMode::Pipeline));
                        return true;
                    }
                    return false;
                });
        }
    };
}