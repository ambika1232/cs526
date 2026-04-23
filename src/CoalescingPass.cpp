#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Casting.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/InstIterator.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"

using namespace llvm;

namespace
{

  enum class ThreadVarKind
  {
    None,
    TidX,
    TidY
  };

  enum class TransformKind
  {
    KeepGlobal,     // already good
    BroadcastReuse, // invariant across x load
    TileRemap,      // strided / bad pattern
    SkipUnknown     // not enough confidence
  };

  struct AccessInfo
  {
    std::string Kind;
    std::string BaseName;
    std::string BaseValueStr;
    std::string OffsetStr;
    bool ThreadDependent = false;
    int64_t StrideXBytes = 0;
    uint64_t AccessSize = 0;
    bool Exact = false;
    bool Contiguous = false;
    bool Monotonic = false;
    bool Broadcast = false;
    std::string ClassName;
    std::string Suggestion;

    TransformKind Action = TransformKind::SkipUnknown;
    std::string ActionReason;
  };

  static const char *transformKindName(TransformKind K)
  {
    switch (K)
    {
    case TransformKind::KeepGlobal:
      return "KEEP_GLOBAL";
    case TransformKind::BroadcastReuse:
      return "BROADCAST_REUSE";
    case TransformKind::TileRemap:
      return "TILE_REMAP";
    case TransformKind::SkipUnknown:
      return "SKIP_UNKNOWN";
    }
    return "SKIP_UNKNOWN";
  }

  static bool shouldSkipFunction(const Function &F)
  {
    if (F.isDeclaration())
      return true;
    // return F.getName().startswith("__clang_ocl_kern_imp_");
    return F.getName().starts_with("__clang_ocl_kern_imp_");
  }

  static const Value *stripPointerCastsAndOffsets(const Value *V)
  {
    while (true)
    {
      if (auto *BC = dyn_cast<BitCastOperator>(V))
      {
        V = BC->getOperand(0);
        continue;
      }
      if (auto *ASC = dyn_cast<AddrSpaceCastOperator>(V))
      {
        V = ASC->getOperand(0);
        continue;
      }
      if (auto *GEP = dyn_cast<GEPOperator>(V))
      {
        V = GEP->getPointerOperand();
        continue;
      }
      if (auto *I = dyn_cast<Instruction>(V))
      {
        if (auto *BCI = dyn_cast<BitCastInst>(I))
        {
          V = BCI->getOperand(0);
          continue;
        }
        if (auto *ASCI = dyn_cast<AddrSpaceCastInst>(I))
        {
          V = ASCI->getOperand(0);
          continue;
        }
        if (auto *GEPI = dyn_cast<GetElementPtrInst>(I))
        {
          V = GEPI->getPointerOperand();
          continue;
        }
      }
    }
    return V;
  }

  static std::string valueToString(const Value *V)
  {
    std::string S;
    raw_string_ostream OS(S);
    if (V->hasName())
      OS << V->getName();
    else
      V->printAsOperand(OS, false);
    return OS.str();
  }

  static std::string getBaseName(const Value *Ptr)
  {
    const Value *Base = stripPointerCastsAndOffsets(Ptr);
    if (auto *A = dyn_cast<Argument>(Base))
    {
      if (A->hasName())
        return A->getName().str();
      return "arg" + std::to_string(A->getArgNo());
    }
    return valueToString(Base);
  }

  static std::string suggestOptimization(const AccessInfo &AI)
  {
    if (AI.ClassName == "INVARIANT_ACROSS_X")
    {
      if (AI.Kind == "load")
        return "candidate: shared-memory/local-memory preload; reuse across x threads";
      return "candidate: no coalescing issue; check write placement";
    }

    if (AI.ClassName == "FULLY_COALESCED" ||
        AI.ClassName == "COALESCED_BUT_MISALIGNED" ||
        AI.ClassName == "PARTIALLY_COALESCED" ||
        AI.ClassName == "LIKELY_COALESCED")
    {
      if (AI.Kind == "load")
        return "candidate: keep as global coalesced load";
      return "candidate: keep as coalesced store";
    }

    if (AI.ThreadDependent &&
        std::llabs(AI.StrideXBytes) > static_cast<int64_t>(AI.AccessSize))
      return "candidate: tile/remap through shared memory to enforce contiguous x access";

    if (AI.ThreadDependent && AI.StrideXBytes == 0)
      return "candidate: broadcast/reuse analysis; consider hoisting or local caching";

    return "candidate: inspect with SCEV for affine decomposition";
  }

} // namespace

struct AffineExpr
{
  bool Known = true;
  bool HasUnknownInvariant = false;

  int64_t CoeffTidX = 0;
  int64_t CoeffTidY = 0;
  int64_t Constant = 0;

  bool dependsOnThreads() const
  {
    return CoeffTidX != 0 || CoeffTidY != 0;
  }
};

struct SCEVAffineSummary
{
  bool Known = true;
  int64_t CoeffTidX = 0;
  int64_t CoeffTidY = 0;
  int64_t Constant = 0;
  bool HasOtherSymbolic = false;
  bool HasLoopRecurrence = false;
};

static AffineExpr invalidExpr()
{
  AffineExpr E;
  E.Known = false;
  return E;
}

static AffineExpr makeConst(int64_t C)
{
  AffineExpr E;
  E.Known = true;
  E.Constant = C;
  return E;
}

static AffineExpr makeUnknownInvariant()
{
  AffineExpr E;
  E.Known = true;
  E.HasUnknownInvariant = true;
  return E;
}

static AffineExpr makeTidX()
{
  AffineExpr E;
  E.Known = true;
  E.CoeffTidX = 1;
  return E;
}

static AffineExpr makeTidY()
{
  AffineExpr E;
  E.Known = true;
  E.CoeffTidY = 1;
  return E;
}

static Value *stripSimpleCasts(Value *V)
{
  while (true)
  {
    if (auto *CI = dyn_cast<CastInst>(V))
    {
      V = CI->getOperand(0);
      continue;
    }
    if (auto *CE = dyn_cast<ConstantExpr>(V))
    {
      if (CE->isCast())
      {
        V = CE->getOperand(0);
        continue;
      }
    }
    break;
  }
  return V;
}

static bool getConstInt(Value *V, int64_t &C)
{
  V = stripSimpleCasts(V);
  if (auto *CI = dyn_cast<ConstantInt>(V))
  {
    C = CI->getSExtValue();
    return true;
  }
  return false;
}

static Value *resolveAllocaStoredValue(Value *V)
{
  V = stripSimpleCasts(V);

  auto *LI = dyn_cast<LoadInst>(V);
  if (!LI)
    return V;

  Value *Ptr = LI->getPointerOperand();
  Ptr = stripSimpleCasts(Ptr);

  auto *AI = dyn_cast<AllocaInst>(Ptr);
  if (!AI)
    return V;

  Value *Stored = nullptr;

  for (User *U : AI->users())
  {
    auto *SI = dyn_cast<StoreInst>(U);
    if (!SI)
      continue;

    if (SI->getPointerOperand() != AI)
      continue;

    if (Stored)
      return V; // multiple stores, give up conservatively

    Stored = SI->getValueOperand();
  }

  if (!Stored)
    return V;

  return stripSimpleCasts(Stored);
}

static ThreadVarKind getThreadVarKind(Value *V)
{
  V = stripSimpleCasts(V);

  V = resolveAllocaStoredValue(V);

  auto *CB = dyn_cast<CallBase>(V);
  if (!CB)
    return ThreadVarKind::None;

  Function *F = CB->getCalledFunction();
  if (!F)
    return ThreadVarKind::None;

  if (!F->getName().contains("get_global_id"))
    return ThreadVarKind::None;

  if (CB->arg_size() < 1)
    return ThreadVarKind::None;

  int64_t Dim = -1;
  if (!getConstInt(CB->getArgOperand(0), Dim))
    return ThreadVarKind::None;

  if (Dim == 0)
    return ThreadVarKind::TidX;
  if (Dim == 1)
    return ThreadVarKind::TidY;

  return ThreadVarKind::None;
}

static bool isInvariantButUnknown(Value *V);

static const char *affineThreadKindName(const AffineExpr &E)
{
  if (!E.Known)
    return "unknown";

  bool HasX = (E.CoeffTidX != 0);
  bool HasY = (E.CoeffTidY != 0);

  if (HasX && !HasY)
    return "affine_tid_x";
  if (!HasX && HasY)
    return "affine_tid_y";
  if (HasX && HasY)
    return "affine_tid_xy";

  if (E.HasUnknownInvariant)
    return "invariant_sym";

  return "other";
}

static Value *findGlobalIdDim(Function &F, int64_t TargetDim)
{
  for (Instruction &I : instructions(F))
  {
    auto *CB = dyn_cast<CallBase>(&I);
    if (!CB)
      continue;

    Function *Callee = CB->getCalledFunction();
    if (!Callee)
      continue;

    if (!Callee->getName().contains("get_global_id"))
      continue;

    if (CB->arg_size() < 1)
      continue;

    int64_t Dim = -1;
    if (!getConstInt(CB->getArgOperand(0), Dim))
      continue;

    if (Dim == TargetDim)
      return CB;
  }

  return nullptr;
}

static Value *findTidXValue(Function &F)
{
  return findGlobalIdDim(F, 0);
}

static Value *findTidYValue(Function &F)
{
  return findGlobalIdDim(F, 1);
}

static std::string formatAffineExpr(const AffineExpr &E)
{
  if (!E.Known)
    return "unknown";

  std::string S;
  bool First = true;

  auto appendTerm = [&](int64_t Coeff, StringRef Name)
  {
    if (Coeff == 0)
      return;

    std::string Term;
    if (Coeff == 1)
      Term = Name.str();
    else if (Coeff == -1)
      Term = "-" + Name.str();
    else
      Term = std::to_string(Coeff) + "*" + Name.str();

    if (First)
    {
      S += Term;
      First = false;
    }
    else
    {
      if (Coeff > 0)
        S += "+" + Term;
      else
        S += Term;
    }
  };

  appendTerm(E.CoeffTidX, "tid_x");
  appendTerm(E.CoeffTidY, "tid_y");

  if (E.Constant != 0)
  {
    if (First)
    {
      S += std::to_string(E.Constant);
      First = false;
    }
    else if (E.Constant > 0)
    {
      S += "+" + std::to_string(E.Constant);
    }
    else
    {
      S += std::to_string(E.Constant);
    }
  }
  else if (First && !E.HasUnknownInvariant)
  {
    S += "0";
    First = false;
  }

  if (E.HasUnknownInvariant)
  {
    if (First)
      S += "sym";
    else
      S += "+sym";
  }

  return S;
}

static AffineExpr combineAdd(const AffineExpr &A, const AffineExpr &B)
{
  if (!A.Known || !B.Known)
    return invalidExpr();

  AffineExpr E;
  E.Known = true;
  E.CoeffTidX = A.CoeffTidX + B.CoeffTidX;
  E.CoeffTidY = A.CoeffTidY + B.CoeffTidY;
  E.Constant = A.Constant + B.Constant;
  E.HasUnknownInvariant = A.HasUnknownInvariant || B.HasUnknownInvariant;
  return E;
}

static AffineExpr scaleExpr(const AffineExpr &A, int64_t Scale)
{
  if (!A.Known)
    return invalidExpr();

  AffineExpr E;
  E.Known = true;
  E.CoeffTidX = A.CoeffTidX * Scale;
  E.CoeffTidY = A.CoeffTidY * Scale;
  E.Constant = A.Constant * Scale;
  E.HasUnknownInvariant = A.HasUnknownInvariant;
  return E;
}

static AffineExpr divideExprIfPossible(const AffineExpr &A, int64_t Div)
{
  if (!A.Known || Div == 0)
    return invalidExpr();

  if (A.HasUnknownInvariant)
    return invalidExpr();

  if ((A.CoeffTidX % Div) != 0 || (A.CoeffTidY % Div) != 0 ||
      (A.Constant % Div) != 0)
    return invalidExpr();

  AffineExpr E;
  E.Known = true;
  E.CoeffTidX = A.CoeffTidX / Div;
  E.CoeffTidY = A.CoeffTidY / Div;
  E.Constant = A.Constant / Div;
  return E;
}

static bool isLoopPHI(Value *V)
{
  auto *PHI = dyn_cast<PHINode>(V);
  if (!PHI)
    return false;

  for (Value *Incoming : PHI->incoming_values())
  {
    Incoming = stripSimpleCasts(Incoming);

    if (Incoming == PHI)
      continue;

    if (isa<ConstantInt>(Incoming))
      continue;

    if (auto *BO = dyn_cast<BinaryOperator>(Incoming))
    {
      if (BO->getOpcode() == Instruction::Add ||
          BO->getOpcode() == Instruction::Sub)
      {
        bool UsesSelf = false;
        bool HasConst = false;

        for (Value *Op : BO->operands())
        {
          Op = stripSimpleCasts(Op);
          if (Op == PHI)
            UsesSelf = true;

          int64_t C = 0;
          if (getConstInt(Op, C))
            HasConst = true;
        }

        if (UsesSelf && HasConst)
          continue;
      }
    }

    return false;
  }

  return true;
}

static bool isInvariantButUnknown(Value *V)
{
  V = stripSimpleCasts(V);

  if (isa<ConstantInt>(V))
    return true;

  switch (getThreadVarKind(V))
  {
  case ThreadVarKind::TidX:
  case ThreadVarKind::TidY:
    return false;
  case ThreadVarKind::None:
    break;
  }

  if (isa<Argument>(V))
    return true;

  if (isa<LoadInst>(V))
    return true;

  if (isLoopPHI(V))
    return true;

  if (auto *BO = dyn_cast<BinaryOperator>(V))
  {
    return isInvariantButUnknown(BO->getOperand(0)) &&
           isInvariantButUnknown(BO->getOperand(1));
  }

  if (auto *PHI = dyn_cast<PHINode>(V))
  {
    (void)PHI;
    return isLoopPHI(V);
  }

  if (auto *SI = dyn_cast<SelectInst>(V))
  {
    return isInvariantButUnknown(SI->getTrueValue()) &&
           isInvariantButUnknown(SI->getFalseValue());
  }

  if (auto *CB = dyn_cast<CallBase>(V))
  {
    (void)CB;
    return false;
  }

  return false;
}

static AffineExpr parseAffine(Value *V)
{
  V = stripSimpleCasts(V);
  V = resolveAllocaStoredValue(V);

  switch (getThreadVarKind(V))
  {
  case ThreadVarKind::TidX:
    return makeTidX();
  case ThreadVarKind::TidY:
    return makeTidY();
  case ThreadVarKind::None:
    break;
  }

  if (auto *CI = dyn_cast<ConstantInt>(V))
    return makeConst(CI->getSExtValue());

  if (isa<Argument>(V))
    return makeUnknownInvariant();

  if (isa<LoadInst>(V))
    return makeUnknownInvariant();

  if (isLoopPHI(V))
    return makeUnknownInvariant();

  if (auto *SI = dyn_cast<SelectInst>(V))
  {
    AffineExpr T = parseAffine(SI->getTrueValue());
    AffineExpr F = parseAffine(SI->getFalseValue());

    if (T.Known && F.Known &&
        T.CoeffTidX == F.CoeffTidX &&
        T.CoeffTidY == F.CoeffTidY)
    {
      AffineExpr E;
      E.Known = true;
      E.CoeffTidX = T.CoeffTidX;
      E.CoeffTidY = T.CoeffTidY;
      E.HasUnknownInvariant = true;
      return E;
    }

    if (isInvariantButUnknown(V))
    {
      V = stripSimpleCasts(V);
      V = resolveAllocaStoredValue(V);
      return makeUnknownInvariant();
    }
    return invalidExpr();
  }

  auto *BO = dyn_cast<BinaryOperator>(V);
  if (!BO)
  {
    if (isInvariantButUnknown(V))
      return makeUnknownInvariant();
    return invalidExpr();
  }

  switch (BO->getOpcode())
  {
  case Instruction::Add:
  {
    AffineExpr L = parseAffine(BO->getOperand(0));
    AffineExpr R = parseAffine(BO->getOperand(1));
    return combineAdd(L, R);
  }

  case Instruction::Sub:
  {
    AffineExpr L = parseAffine(BO->getOperand(0));
    AffineExpr R = parseAffine(BO->getOperand(1));
    if (!L.Known || !R.Known)
      return invalidExpr();
    return combineAdd(L, scaleExpr(R, -1));
  }

  case Instruction::Mul:
  {
    int64_t C = 0;

    AffineExpr L = parseAffine(BO->getOperand(0));
    if (getConstInt(BO->getOperand(1), C) && L.Known)
      return scaleExpr(L, C);

    AffineExpr R = parseAffine(BO->getOperand(1));
    if (getConstInt(BO->getOperand(0), C) && R.Known)
      return scaleExpr(R, C);

    bool LopInvariantUnknown = isInvariantButUnknown(BO->getOperand(0));
    bool RopInvariantUnknown = isInvariantButUnknown(BO->getOperand(1));

    if (L.Known && !L.HasUnknownInvariant && L.CoeffTidX == 0 &&
        RopInvariantUnknown)
    {
      return makeUnknownInvariant();
    }

    if (R.Known && !R.HasUnknownInvariant && R.CoeffTidX == 0 &&
        LopInvariantUnknown)
    {
      return makeUnknownInvariant();
    }

    if (LopInvariantUnknown && RopInvariantUnknown)
      return makeUnknownInvariant();

    return invalidExpr();
  }

  case Instruction::Shl:
  {
    int64_t ShiftAmt = 0;
    if (!getConstInt(BO->getOperand(1), ShiftAmt))
      return invalidExpr();
    if (ShiftAmt < 0 || ShiftAmt >= 62)
      return invalidExpr();

    AffineExpr X = parseAffine(BO->getOperand(0));
    if (!X.Known)
      return invalidExpr();

    int64_t Scale = 1LL << ShiftAmt;
    return scaleExpr(X, Scale);
  }

  case Instruction::AShr:
  case Instruction::LShr:
  {
    int64_t ShiftAmt = 0;
    if (!getConstInt(BO->getOperand(1), ShiftAmt))
      return invalidExpr();
    if (ShiftAmt < 0 || ShiftAmt >= 62)
      return invalidExpr();

    AffineExpr X = parseAffine(BO->getOperand(0));
    if (!X.Known)
      return invalidExpr();

    int64_t Div = 1LL << ShiftAmt;
    return divideExprIfPossible(X, Div);
  }

  default:
    if (isInvariantButUnknown(V))
      return makeUnknownInvariant();
    return invalidExpr();
  }
}

static uint64_t getTypeSizeInBytes(const DataLayout &DL, Type *Ty)
{
  if (!Ty || !Ty->isSized())
    return 0;
  return DL.getTypeAllocSize(Ty);
}

static AffineExpr buildByteOffsetExpr(GetElementPtrInst *GEP,
                                      const DataLayout &DL)
{
  AffineExpr Total = makeConst(0);

  for (auto GTI = llvm::gep_type_begin(GEP), GTE = llvm::gep_type_end(GEP);
       GTI != GTE; ++GTI)
  {
    Value *Idx = GTI.getOperand();

    if (StructType *STy = GTI.getStructTypeOrNull())
    {
      int64_t FieldNo = -1;
      if (!getConstInt(Idx, FieldNo))
        return invalidExpr();

      if (FieldNo < 0 ||
          static_cast<unsigned>(FieldNo) >= STy->getNumElements())
        return invalidExpr();

      const StructLayout *SL = DL.getStructLayout(STy);
      uint64_t FieldOffset =
          SL->getElementOffset(static_cast<unsigned>(FieldNo));
      Total =
          combineAdd(Total, makeConst(static_cast<int64_t>(FieldOffset)));
      continue;
    }

    Type *IndexedTy = GTI.getIndexedType();
    if (!IndexedTy)
      return invalidExpr();

    uint64_t ElemSize = getTypeSizeInBytes(DL, IndexedTy);
    if (ElemSize == 0)
      return invalidExpr();

    AffineExpr IdxExpr = parseAffine(Idx);
    if (!IdxExpr.Known)
      return invalidExpr();

    Total =
        combineAdd(Total, scaleExpr(IdxExpr, static_cast<int64_t>(ElemSize)));
  }

  return Total;
}

static AffineExpr buildByteOffsetExprRecursive(Value *Ptr,
                                               const DataLayout &DL)
{
  Ptr = stripSimpleCasts(Ptr);

  if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr))
  {
    AffineExpr Here = buildByteOffsetExpr(GEP, DL);
    AffineExpr Base = buildByteOffsetExprRecursive(GEP->getPointerOperand(), DL);

    if (!Here.Known || !Base.Known)
      return invalidExpr();

    return combineAdd(Base, Here);
  }

  if (auto *GEPOp = dyn_cast<GEPOperator>(Ptr))
  {
    AffineExpr Total = makeConst(0);

    for (auto GTI = gep_type_begin(GEPOp), GTE = gep_type_end(GEPOp);
         GTI != GTE; ++GTI)
    {
      Value *Idx = GTI.getOperand();

      if (StructType *STy = GTI.getStructTypeOrNull())
      {
        int64_t FieldNo = -1;
        if (!getConstInt(Idx, FieldNo))
          return invalidExpr();

        if (FieldNo < 0 ||
            static_cast<unsigned>(FieldNo) >= STy->getNumElements())
          return invalidExpr();

        const StructLayout *SL = DL.getStructLayout(STy);
        uint64_t FieldOffset =
            SL->getElementOffset(static_cast<unsigned>(FieldNo));
        Total = combineAdd(Total,
                           makeConst(static_cast<int64_t>(FieldOffset)));
        continue;
      }

      Type *IndexedTy = GTI.getIndexedType();
      if (!IndexedTy)
        return invalidExpr();

      uint64_t ElemSize = getTypeSizeInBytes(DL, IndexedTy);
      if (ElemSize == 0)
        return invalidExpr();

      AffineExpr IdxExpr = parseAffine(Idx);
      if (!IdxExpr.Known)
        return invalidExpr();

      Total =
          combineAdd(Total, scaleExpr(IdxExpr, static_cast<int64_t>(ElemSize)));
    }

    AffineExpr Base =
        buildByteOffsetExprRecursive(const_cast<Value *>(GEPOp->getPointerOperand()), DL);
    if (!Base.Known || !Total.Known)
      return invalidExpr();

    return combineAdd(Base, Total);
  }

  // Base pointer argument/global: zero extra offset at this level.
  return makeConst(0);
}

static bool canEvaluateExactly(const AffineExpr &E)
{
  return E.Known && !E.HasUnknownInvariant;
}

static int64_t evaluateAffineBytes(const AffineExpr &E, int64_t TidX,
                                   int64_t TidY)
{
  return E.CoeffTidX * TidX + E.CoeffTidY * TidY + E.Constant;
}

static int64_t computeStrideXBytes(const AffineExpr &E)
{
  if (!E.Known)
    return 0;
  return std::llabs(E.CoeffTidX);
}

static bool getConstFromSCEV(const SCEV *S, int64_t &Out)
{
  if (auto *C = dyn_cast<SCEVConstant>(S))
  {
    Out = C->getAPInt().getSExtValue();
    return true;
  }
  return false;
}

static SCEVAffineSummary summarizeSCEV(const SCEV *S,
                                       const SCEV *TidXS,
                                       const SCEV *TidYS)
{
  SCEVAffineSummary R;

  if (auto *SC = dyn_cast<SCEVSignExtendExpr>(S))
    return summarizeSCEV(SC->getOperand(), TidXS, TidYS);

  if (auto *ZC = dyn_cast<SCEVZeroExtendExpr>(S))
    return summarizeSCEV(ZC->getOperand(), TidXS, TidYS);

  if (auto *TC = dyn_cast<SCEVTruncateExpr>(S))
    return summarizeSCEV(TC->getOperand(), TidXS, TidYS);

  if (auto *C = dyn_cast<SCEVConstant>(S))
  {
    R.Constant = C->getAPInt().getSExtValue();
    return R;
  }

  if (TidXS && S == TidXS)
  {
    R.CoeffTidX = 1;
    return R;
  }

  if (TidYS && S == TidYS)
  {
    R.CoeffTidY = 1;
    return R;
  }

  if (auto *Add = dyn_cast<SCEVAddExpr>(S))
  {
    SCEVAffineSummary Total;
    for (const SCEV *Op : Add->operands())
    {
      SCEVAffineSummary Part = summarizeSCEV(Op, TidXS, TidYS);
      if (!Part.Known)
      {
        Total.Known = false;
        return Total;
      }

      Total.CoeffTidX += Part.CoeffTidX;
      Total.CoeffTidY += Part.CoeffTidY;
      Total.Constant += Part.Constant;
      Total.HasOtherSymbolic |= Part.HasOtherSymbolic;
      Total.HasLoopRecurrence |= Part.HasLoopRecurrence;
    }
    return Total;
  }

  if (auto *Mul = dyn_cast<SCEVMulExpr>(S))
  {
    if (Mul->getNumOperands() != 2)
    {
      R.Known = false;
      return R;
    }

    const SCEV *A = Mul->getOperand(0);
    const SCEV *B = Mul->getOperand(1);
    int64_t C = 0;

    if (getConstFromSCEV(A, C))
    {
      SCEVAffineSummary Part = summarizeSCEV(B, TidXS, TidYS);
      if (!Part.Known || Part.HasOtherSymbolic || Part.HasLoopRecurrence)
      {
        R.Known = false;
        return R;
      }

      Part.CoeffTidX *= C;
      Part.CoeffTidY *= C;
      Part.Constant *= C;
      return Part;
    }

    if (getConstFromSCEV(B, C))
    {
      SCEVAffineSummary Part = summarizeSCEV(A, TidXS, TidYS);
      if (!Part.Known || Part.HasOtherSymbolic || Part.HasLoopRecurrence)
      {
        R.Known = false;
        return R;
      }

      Part.CoeffTidX *= C;
      Part.CoeffTidY *= C;
      Part.Constant *= C;
      return Part;
    }

    R.Known = false;
    return R;
  }

  if (isa<SCEVAddRecExpr>(S))
  {
    R.HasLoopRecurrence = true;
    return R;
  }

  if (auto *U = dyn_cast<SCEVUnknown>(S))
  {
    Value *V = const_cast<Value *>(U->getValue());
    V = stripSimpleCasts(V);
    V = resolveAllocaStoredValue(V);

    switch (getThreadVarKind(V))
    {
    case ThreadVarKind::TidX:
      R.CoeffTidX = 1;
      return R;
    case ThreadVarKind::TidY:
      R.CoeffTidY = 1;
      return R;
    case ThreadVarKind::None:
      break;
    }

    if (isa<ConstantInt>(V))
    {
      if (auto *CI = dyn_cast<ConstantInt>(V))
      {
        R.Constant = CI->getSExtValue();
        return R;
      }
    }

    R.HasOtherSymbolic = true;
    return R;
  }
  R.Known = false;
  return R;
}

struct WarpAccessInfo
{
  bool Valid = false;
  bool Broadcast = false;
  bool Contiguous = false;
  bool Monotonic = false;
  bool Aligned128 = false;
  bool Exact = false;
  unsigned Num128BTransactions = 0;
  int64_t FirstByte = 0;
  int64_t LastByte = 0;
};

static WarpAccessInfo analyzeWarp(const AffineExpr &ByteExpr,
                                  uint64_t AccessSizeBytes)
{
  WarpAccessInfo Info;
  if (!ByteExpr.Known || AccessSizeBytes == 0)
    return Info;

  Info.Valid = true;
  Info.Exact = canEvaluateExactly(ByteExpr);

  if (!Info.Exact)
  {
    int64_t Stride = computeStrideXBytes(ByteExpr);
    Info.Broadcast = (Stride == 0);
    Info.Contiguous = (Stride == static_cast<int64_t>(AccessSizeBytes));
    Info.Monotonic = (ByteExpr.CoeffTidX >= 0);
    return Info;
  }

  std::vector<int64_t> Addrs;
  Addrs.reserve(32);

  for (int Lane = 0; Lane < 32; ++Lane)
    Addrs.push_back(evaluateAffineBytes(ByteExpr, Lane, 0));

  Info.FirstByte = Addrs.front();
  Info.LastByte = Addrs.back();

  Info.Broadcast =
      std::all_of(Addrs.begin(), Addrs.end(),
                  [&](int64_t A)
                  { return A == Addrs.front(); });

  Info.Monotonic = true;
  for (size_t I = 1; I < Addrs.size(); ++I)
  {
    if (Addrs[I] < Addrs[I - 1])
    {
      Info.Monotonic = false;
      break;
    }
  }

  Info.Contiguous = true;
  for (size_t I = 1; I < Addrs.size(); ++I)
  {
    if (Addrs[I] != Addrs[I - 1] + static_cast<int64_t>(AccessSizeBytes))
    {
      Info.Contiguous = false;
      break;
    }
  }

  Info.Aligned128 = ((Addrs.front() % 128) == 0);

  std::set<int64_t> Segments;
  for (int Lane = 0; Lane < 32; ++Lane)
  {
    int64_t Start = Addrs[Lane];
    int64_t End = Start + static_cast<int64_t>(AccessSizeBytes) - 1;
    int64_t FirstSeg = Start / 128;
    int64_t LastSeg = End / 128;
    for (int64_t Seg = FirstSeg; Seg <= LastSeg; ++Seg)
      Segments.insert(Seg);
  }
  Info.Num128BTransactions = static_cast<unsigned>(Segments.size());
  return Info;
}

static std::string classifyAccess(const AffineExpr &ByteExpr,
                                  uint64_t AccessSizeBytes,
                                  const WarpAccessInfo &WI)
{
  if (!ByteExpr.Known)
  {
    return "UNKNOWN";
  }

  if (!ByteExpr.dependsOnThreads())
  {
    if (ByteExpr.HasUnknownInvariant)
    {
      return "INVARIANT_ACROSS_X";
    }
    return "THREAD_INVARIANT";
  }

  int64_t StrideBytes = computeStrideXBytes(ByteExpr);

  if (StrideBytes == 0)
  {
    return "INVARIANT_ACROSS_X";
  }

  if (StrideBytes == static_cast<int64_t>(AccessSizeBytes))
  {
    if (WI.Exact)
    {
      if (WI.Num128BTransactions == 1 && WI.Aligned128)
      {
        return "FULLY_COALESCED";
      }
      if (WI.Num128BTransactions <= 2)
      {
        return "COALESCED_BUT_MISALIGNED";
      }
      return "PARTIALLY_COALESCED";
    }
    return "LIKELY_COALESCED";
  }

  if (StrideBytes > static_cast<int64_t>(AccessSizeBytes))
  {
    return "STRIDED";
  }

  if (StrideBytes < static_cast<int64_t>(AccessSizeBytes))
  {
    return "OVERLAPPED_OR_PACKED";
  }

  return "UNKNOWN";
}

static void printAccessSummary(StringRef AccessKind, GetElementPtrInst *GEP,
                               const AffineExpr &ByteExpr,
                               uint64_t AccessSizeBytes)
{
  WarpAccessInfo WI = analyzeWarp(ByteExpr, AccessSizeBytes);

  outs() << "  access=" << AccessKind;

  outs() << " base=";
  GEP->getPointerOperand()->printAsOperand(outs(), false);

  outs() << " byte_offset=" << formatAffineExpr(ByteExpr);
  outs() << " thread_dependent="
         << (ByteExpr.dependsOnThreads() ? "yes" : "no");
  outs() << " stride_x_bytes=" << computeStrideXBytes(ByteExpr);
  outs() << " access_size=" << AccessSizeBytes;
  outs() << " exact=" << (WI.Exact ? "yes" : "no");

  if (WI.Valid && WI.Exact)
  {
    outs() << " tx128=" << WI.Num128BTransactions;
    outs() << " aligned128=" << (WI.Aligned128 ? "yes" : "no");
    outs() << " contiguous=" << (WI.Contiguous ? "yes" : "no");
    outs() << " monotonic=" << (WI.Monotonic ? "yes" : "no");
    outs() << " broadcast=" << (WI.Broadcast ? "yes" : "no");
  }
  else if (WI.Valid)
  {
    outs() << " contiguous=" << (WI.Contiguous ? "yes" : "no");
    outs() << " monotonic=" << (WI.Monotonic ? "yes" : "no");
    outs() << " broadcast=" << (WI.Broadcast ? "yes" : "no");
  }

  outs() << " class="
         << classifyAccess(ByteExpr, AccessSizeBytes, WI);
  outs() << "\n";
}

static TransformKind chooseTransform(const AccessInfo &AI)
{
  // Stores that are already contiguous should be preserved.
  if (AI.Kind == "store" && AI.ClassName == "LIKELY_COALESCED")
    return TransformKind::KeepGlobal;

  // Fully or likely coalesced global accesses are already good.
  if (AI.ClassName == "FULLY_COALESCED" ||
      AI.ClassName == "COALESCED_BUT_MISALIGNED" ||
      AI.ClassName == "PARTIALLY_COALESCED" ||
      AI.ClassName == "LIKELY_COALESCED")
  {
    return TransformKind::KeepGlobal;
  }

  // Invariant loads are better handled by reuse/broadcast staging.
  if (AI.Kind == "load" &&
      (AI.ClassName == "INVARIANT_ACROSS_X" ||
       AI.ClassName == "THREAD_INVARIANT" ||
       AI.Broadcast))
  {
    return TransformKind::BroadcastReuse;
  }

  // Large stride across x is the classic coalescing-rewrite target.
  if (AI.ThreadDependent &&
      AI.AccessSize > 0 &&
      std::llabs(AI.StrideXBytes) > static_cast<int64_t>(AI.AccessSize))
  {
    return TransformKind::TileRemap;
  }

  // If it is thread-dependent but not contiguous, prefer remap.
  if (AI.ThreadDependent && !AI.Contiguous)
    return TransformKind::TileRemap;

  return TransformKind::SkipUnknown;
}

static std::string explainTransformChoice(const AccessInfo &AI)
{
  switch (chooseTransform(AI))
  {
  case TransformKind::KeepGlobal:
    return "access already varies contiguously across tid_x; preserve direct global access";
  case TransformKind::BroadcastReuse:
    return "access is invariant/broadcast across tid_x; stage once and reuse across x threads";
  case TransformKind::TileRemap:
    return "access is thread-dependent but non-contiguous/strided; rewrite through cooperative tiled load";
  case TransformKind::SkipUnknown:
    return "insufficient confidence for a safe profitability-guided rewrite";
  }
  return "insufficient confidence for a safe profitability-guided rewrite";
}

static void dumpIndexDef(Value *V)
{
  V = stripSimpleCasts(V);

  outs() << "      idx_root=";
  V->printAsOperand(outs(), false);
  outs() << "\n";

  if (auto *I = dyn_cast<Instruction>(V))
  {
    outs() << "      idx_opcode=" << I->getOpcodeName() << "\n";

    unsigned OpNo = 0;
    for (Value *Op : I->operands())
    {
      outs() << "        op" << OpNo++ << "=";
      Op->printAsOperand(outs(), false);

      AffineExpr E = parseAffine(Op);
      outs() << " affine=" << formatAffineExpr(E);

      ThreadVarKind TV = getThreadVarKind(Op);
      if (TV == ThreadVarKind::TidX)
        outs() << " kind=tid_x";
      else if (TV == ThreadVarKind::TidY)
        outs() << " kind=tid_y";
      else
        outs() << " kind=" << affineThreadKindName(E);

      int64_t C = 0;
      if (getConstInt(Op, C))
        outs() << " const=" << C;

      outs() << "\n";
    }
  }
}

static void dumpSCEVInfoForIndex(ScalarEvolution &SE,
                                 const SCEV *TidXS,
                                 const SCEV *TidYS,
                                 Value *Idx)
{
  Value *NormIdx = stripSimpleCasts(Idx);
  NormIdx = resolveAllocaStoredValue(NormIdx);

  const SCEV *S = SE.getSCEV(NormIdx);
  SCEVAffineSummary Sum = summarizeSCEVValue(Idx, SE, TidXS, TidYS);

  outs() << "        scev=" << *S
         << " scev_coeff_tid_x=" << Sum.CoeffTidX
         << " scev_coeff_tid_y=" << Sum.CoeffTidY
         << " scev_const=" << Sum.Constant
         << " scev_looprec=" << (Sum.HasLoopRecurrence ? "yes" : "no")
         << " scev_other=" << (Sum.HasOtherSymbolic ? "yes" : "no")
         << " scev_known=" << (Sum.Known ? "yes" : "no")
         << "\n";
}

static std::string classifyLoadForRewrite(LoadInst *LI, const DataLayout &DL)
{
  if (!LI)
    return "UNKNOWN";

  Value *Ptr = LI->getPointerOperand();
  auto *GEP = dyn_cast<GetElementPtrInst>(stripSimpleCasts(Ptr));
  if (!GEP)
    return "UNKNOWN";

  uint64_t AccessSizeBytes = getTypeSizeInBytes(DL, LI->getType());
  if (AccessSizeBytes == 0)
    return "UNKNOWN";

  AffineExpr ByteExpr = buildByteOffsetExprRecursive(Ptr, DL);
  WarpAccessInfo WI = analyzeWarp(ByteExpr, AccessSizeBytes);
  return classifyAccess(ByteExpr, AccessSizeBytes, WI);
}

struct StridedLoadMatch
{
  LoadInst *LI = nullptr;
  GetElementPtrInst *GEP = nullptr;
  Value *BasePtr = nullptr;
  Type *ElemTy = nullptr;

  int64_t StrideElems = 0; // e.g. 4 for A[4*tid]
  int64_t ConstElems = 0;  // constant element offset
  uint64_t ElemSizeBytes = 0;
};

static FunctionCallee getOrCreateOCLBuiltin(Module &M,
                                            StringRef Name,
                                            Type *RetTy,
                                            ArrayRef<Type *> ArgTys)
{
  FunctionType *FTy = FunctionType::get(RetTy, ArgTys, false);
  return M.getOrInsertFunction(Name, FTy);
}

static Value *createGetGlobalId0(IRBuilder<> &B, Module &M)
{
  auto Callee = getOrCreateOCLBuiltin(
      M, "_Z13get_global_idj",
      B.getInt64Ty(),
      {B.getInt32Ty()});
  return B.CreateCall(Callee, {B.getInt32(0)}, "gid0");
}

static Value *createGetLocalId0(IRBuilder<> &B, Module &M)
{
  auto Callee = getOrCreateOCLBuiltin(
      M, "_Z12get_local_idj",
      B.getInt64Ty(),
      {B.getInt32Ty()});
  return B.CreateCall(Callee, {B.getInt32(0)}, "lid0");
}

static Value *createGetLocalSize0(IRBuilder<> &B, Module &M)
{
  auto Callee = getOrCreateOCLBuiltin(
      M, "_Z14get_local_sizej",
      B.getInt64Ty(),
      {B.getInt32Ty()});
  return B.CreateCall(Callee, {B.getInt32(0)}, "lsize0");
}

static CallInst *createBarrier(IRBuilder<> &B, Module &M)
{
  auto Callee = getOrCreateOCLBuiltin(
      M, "_Z7barrierj",
      B.getVoidTy(),
      {B.getInt32Ty()});

  // CLK_LOCAL_MEM_FENCE = 1
  return B.CreateCall(Callee, {B.getInt32(1)});
}

struct SCEVGEPIndexInfo
{
  bool Known = false;
  int64_t CoeffTidX = 0;
  int64_t CoeffTidY = 0;
  int64_t ConstantElems = 0;
  bool HasOtherSymbolic = false;
  bool HasLoopRecurrence = false;
};

static SCEVAffineSummary summarizeSCEVValue(Value *V,
                                            ScalarEvolution &SE,
                                            const SCEV *TidXS,
                                            const SCEV *TidYS)
{
  Value *NormV = stripSimpleCasts(V);
  NormV = resolveAllocaStoredValue(NormV);

  switch (getThreadVarKind(NormV))
  {
  case ThreadVarKind::TidX:
  {
    SCEVAffineSummary R;
    R.CoeffTidX = 1;
    return R;
  }
  case ThreadVarKind::TidY:
  {
    SCEVAffineSummary R;
    R.CoeffTidY = 1;
    return R;
  }
  case ThreadVarKind::None:
    break;
  }

  const SCEV *S = SE.getSCEV(NormV);
  return summarizeSCEV(S, TidXS, TidYS);
}

static bool summarizeSimpleGEPIndex(GetElementPtrInst *GEP,
                                    ScalarEvolution &SE,
                                    const SCEV *TidXS,
                                    const SCEV *TidYS,
                                    SCEVGEPIndexInfo &Out)
{
  if (!GEP)
    return false;

  // For now only handle a single non-struct index cleanly.
  Value *OnlyIdx = nullptr;
  Type *IndexedTy = nullptr;

  for (auto GTI = gep_type_begin(GEP), GTE = gep_type_end(GEP); GTI != GTE; ++GTI)
  {
    Value *Idx = GTI.getOperand();

    if (GTI.getStructTypeOrNull())
      return false; // skip struct indexing for now

    if (OnlyIdx)
      return false; // multiple non-struct indices: not handling yet

    OnlyIdx = Idx;
    IndexedTy = GTI.getIndexedType();
  }

  if (!OnlyIdx || !IndexedTy)
    return false;

  SCEVAffineSummary Sum = summarizeSCEVValue(OnlyIdx, SE, TidXS, TidYS);

  Out.Known = Sum.Known;
  Out.CoeffTidX = Sum.CoeffTidX;
  Out.CoeffTidY = Sum.CoeffTidY;
  Out.ConstantElems = Sum.Constant;
  Out.HasOtherSymbolic = Sum.HasOtherSymbolic;
  Out.HasLoopRecurrence = Sum.HasLoopRecurrence;

  return true;
}

static bool matchSimpleStridedLoadSCEV(LoadInst *LI,
                                       const DataLayout &DL,
                                       ScalarEvolution &SE,
                                       const SCEV *TidXS,
                                       const SCEV *TidYS,
                                       StridedLoadMatch &Out)
{
  if (!LI || LI->isVolatile() || LI->isAtomic())
    return false;

  Value *Ptr = LI->getPointerOperand();
  auto *GEP = dyn_cast<GetElementPtrInst>(stripSimpleCasts(Ptr));
  if (!GEP)
    return false;

  Type *ElemTy = LI->getType();
  uint64_t ElemSize = getTypeSizeInBytes(DL, ElemTy);
  if (ElemSize == 0)
    return false;

  SCEVGEPIndexInfo GI;
  if (!summarizeSimpleGEPIndex(GEP, SE, TidXS, TidYS, GI))
    return false;

  if (!GI.Known)
    return false;

  if (GI.HasOtherSymbolic || GI.HasLoopRecurrence)
    return false;

  // Only 1D tid_x patterns for now.
  if (GI.CoeffTidY != 0)
    return false;

  if (GI.CoeffTidX == 0)
    return false;

  int64_t StrideElems = GI.CoeffTidX;
  int64_t ConstElems = GI.ConstantElems;

  if (StrideElems <= 1)
    return false;

  Out.LI = LI;
  Out.GEP = GEP;
  Out.BasePtr = GEP->getPointerOperand();
  Out.ElemTy = ElemTy;
  Out.StrideElems = StrideElems;
  Out.ConstElems = ConstElems;
  Out.ElemSizeBytes = ElemSize;

  outs() << "[SCEV strided match] ";
  LI->printAsOperand(outs(), false);
  outs() << " stride_elems=" << StrideElems
         << " const_elems=" << ConstElems
         << "\n";

  return true;
}

static bool matchSimpleStridedLoad(LoadInst *LI,
                                   const DataLayout &DL,
                                   StridedLoadMatch &Out)
{
  if (!LI || LI->isVolatile() || LI->isAtomic())
    return false;

  Value *Ptr = LI->getPointerOperand();
  auto *GEP = dyn_cast<GetElementPtrInst>(stripSimpleCasts(Ptr));
  if (!GEP)
    return false;

  Type *ElemTy = LI->getType();
  uint64_t ElemSize = getTypeSizeInBytes(DL, ElemTy);
  if (ElemSize == 0)
    return false;

  AffineExpr ByteExpr = buildByteOffsetExprRecursive(Ptr, DL);
  if (!ByteExpr.Known || ByteExpr.HasUnknownInvariant)
    return false;

  // Only 1D tid_x patterns for now.
  if (ByteExpr.CoeffTidY != 0)
    return false;

  // Needs to be thread-dependent, strided, and exact in element units.
  if (ByteExpr.CoeffTidX == 0)
    return false;

  if ((ByteExpr.CoeffTidX % static_cast<int64_t>(ElemSize)) != 0)
    return false;

  if ((ByteExpr.Constant % static_cast<int64_t>(ElemSize)) != 0)
    return false;

  int64_t StrideElems = ByteExpr.CoeffTidX / static_cast<int64_t>(ElemSize);
  int64_t ConstElems = ByteExpr.Constant / static_cast<int64_t>(ElemSize);

  if (StrideElems <= 1)
    return false;

  WarpAccessInfo WI = analyzeWarp(ByteExpr, ElemSize);
  std::string ClassName = classifyAccess(ByteExpr, ElemSize, WI);
  if (ClassName != "STRIDED")
    return false;

  Out.LI = LI;
  Out.GEP = GEP;
  Out.BasePtr = GEP->getPointerOperand();
  Out.ElemTy = ElemTy;
  Out.StrideElems = StrideElems;
  Out.ConstElems = ConstElems;
  Out.ElemSizeBytes = ElemSize;
  return true;
}
struct CoalescedStoreMatch
{
  StoreInst *SI = nullptr;
  GetElementPtrInst *GEP = nullptr;
  Value *BasePtr = nullptr;
  Value *StoredVal = nullptr;
  Type *ElemTy = nullptr;

  int64_t ConstElems = 0;
  uint64_t ElemSizeBytes = 0;
};

static bool matchSimpleCoalescedStore(StoreInst *SI,
                                      const DataLayout &DL,
                                      CoalescedStoreMatch &Out)
{
  if (!SI || SI->isVolatile() || SI->isAtomic())
    return false;

  Value *Ptr = SI->getPointerOperand();
  auto *GEP = dyn_cast<GetElementPtrInst>(stripSimpleCasts(Ptr));
  if (!GEP)
    return false;

  Type *ElemTy = SI->getValueOperand()->getType();
  uint64_t ElemSize = getTypeSizeInBytes(DL, ElemTy);
  if (ElemSize == 0)
    return false;

  AffineExpr ByteExpr = buildByteOffsetExprRecursive(Ptr, DL);
  if (!ByteExpr.Known || ByteExpr.HasUnknownInvariant)
    return false;

  if (ByteExpr.CoeffTidY != 0)
    return false;

  if (ByteExpr.CoeffTidX != static_cast<int64_t>(ElemSize))
    return false;

  if ((ByteExpr.Constant % static_cast<int64_t>(ElemSize)) != 0)
    return false;

  WarpAccessInfo WI = analyzeWarp(ByteExpr, ElemSize);
  std::string ClassName = classifyAccess(ByteExpr, ElemSize, WI);
  if (!(ClassName == "FULLY_COALESCED" ||
        ClassName == "COALESCED_BUT_MISALIGNED" ||
        ClassName == "PARTIALLY_COALESCED" ||
        ClassName == "LIKELY_COALESCED"))
    return false;

  Out.SI = SI;
  Out.GEP = GEP;
  Out.BasePtr = GEP->getPointerOperand();
  Out.StoredVal = SI->getValueOperand();
  Out.ElemTy = ElemTy;
  Out.ConstElems = ByteExpr.Constant / static_cast<int64_t>(ElemSize);
  Out.ElemSizeBytes = ElemSize;
  return true;
}

struct TileRemapPass : public PassInfoMixin<TileRemapPass>
{
  // PreservedAnalyses run(Function &F, FunctionAnalysisManager &)
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM)
  {

    if (shouldSkipFunction(F))
    {
      errs() << "SKIP function: " << F.getName() << "\n";
      return PreservedAnalyses::all();
    }

    auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
    auto &LI = AM.getResult<LoopAnalysis>(F);
    auto &DT = AM.getResult<DominatorTreeAnalysis>(F);

    Value *TidXVal = findTidXValue(F);
    Value *TidYVal = findTidYValue(F);

    const SCEV *TidXS = TidXVal ? SE.getSCEV(TidXVal) : nullptr;
    const SCEV *TidYS = TidYVal ? SE.getSCEV(TidYVal) : nullptr;

    (void)LI;
    (void)DT;

    Module *M = F.getParent();
    const DataLayout &DL = M->getDataLayout();

    bool Changed = false;

    errs() << "ENTER tile-remap run: " << F.getName() << "\n";

    SmallVector<LoadInst *, 8> CandidateLoads;
    for (BasicBlock &BB : F)
    {
      for (Instruction &I : BB)
      {
        if (auto *LI = dyn_cast<LoadInst>(&I))
        {
          StridedLoadMatch Match;
          if (matchSimpleStridedLoadSCEV(LoadI, DL, SE, TidXS, TidYS, Match))
            CandidateLoads.push_back(LoadI);
        }
      }
    }

    for (LoadInst *LI : CandidateLoads)
    {
      if (!LI->getParent())
        continue;

      StridedLoadMatch Match;
      if (!matchSimpleStridedLoadSCEV(LI, DL, SE, TidXS, TidYS, Match))
        continue;

      BasicBlock *OrigBB = LI->getParent();
      BasicBlock *ContBB = OrigBB->splitBasicBlock(LI, "tile.cont");
      LLVMContext &Ctx = F.getContext();

      // Preload loop blocks.
      BasicBlock *HeaderBB =
          BasicBlock::Create(Ctx, "tile.preload.header", &F, ContBB);
      BasicBlock *BodyBB =
          BasicBlock::Create(Ctx, "tile.preload.body", &F, ContBB);
      BasicBlock *ExitBB =
          BasicBlock::Create(Ctx, "tile.preload.exit", &F, ContBB);

      // Replace the auto branch created by splitBasicBlock.
      OrigBB->getTerminator()->eraseFromParent();
      IRBuilder<> BOrig(OrigBB);

      Value *Lid0 = createGetLocalId0(BOrig, *M);
      Value *LSize0 = createGetLocalSize0(BOrig, *M);
      Value *Gid0 = createGetGlobalId0(BOrig, *M);

      // groupBase = gid - lid
      Value *GroupBase = BOrig.CreateSub(Gid0, Lid0, "group.base");

      // denseBase = stride * groupBase + const
      Value *StrideC = ConstantInt::get(BOrig.getInt64Ty(),
                                        Match.StrideElems);
      Value *ConstC = ConstantInt::get(BOrig.getInt64Ty(),
                                       Match.ConstElems);

      Value *DenseBase = BOrig.CreateAdd(
          BOrig.CreateMul(GroupBase, StrideC, "dense.mul"),
          ConstC,
          "dense.base");

      // Runtime valid span for preload/read.
      Value *ValidTileElems = BOrig.CreateAdd(
          BOrig.CreateMul(
              StrideC,
              BOrig.CreateSub(LSize0,
                              ConstantInt::get(BOrig.getInt64Ty(), 1),
                              "valid.lsize.minus1"),
              "valid.tile.span"),
          ConstantInt::get(BOrig.getInt64Ty(), 1),
          "valid.tile.elems");

      // Fixed allocation size in entry block.
      Value *TileElems = ConstantInt::get(BOrig.getInt64Ty(), 128);

      Instruction *InsertPt = &*F.getEntryBlock().getFirstInsertionPt();
      AllocaInst *Tile = new AllocaInst(
          Match.ElemTy,
          3,
          TileElems,
          Align(4),
          "tile.local",
          InsertPt);

      AllocaInst *OutTile = new AllocaInst(
          Match.ElemTy,
          3,
          TileElems,
          Align(4),
          "tile.out.local",
          InsertPt);

      // // Branch to preload loop.
      // BOrig.CreateBr(HeaderBB);
      BOrig.CreateBr(HeaderBB);

      // Preload header: for (off = lid; off < validTileElems; off += lsize)
      IRBuilder<> BHeader(HeaderBB);
      PHINode *OffPhi = BHeader.CreatePHI(BHeader.getInt64Ty(), 2, "off");
      OffPhi->addIncoming(Lid0, OrigBB);
      Value *Cond = BHeader.CreateICmpULT(OffPhi, ValidTileElems, "tile.cond");
      BHeader.CreateCondBr(Cond, BodyBB, ExitBB);

      // Preload body.
      IRBuilder<> BBody(BodyBB);
      Value *GlobalIdx = BBody.CreateAdd(DenseBase, OffPhi, "global.idx");
      Value *GlobalPtr = BBody.CreateGEP(Match.ElemTy, Match.BasePtr,
                                         GlobalIdx, "global.ptr");
      LoadInst *Loaded = BBody.CreateLoad(Match.ElemTy, GlobalPtr, "tile.ld");

      Value *TilePtr = BBody.CreateGEP(
          Match.ElemTy,
          Tile,
          OffPhi,
          "tile.ptr");
      BBody.CreateStore(Loaded, TilePtr);

      Value *NextOff = BBody.CreateAdd(OffPhi, LSize0, "off.next");
      BBody.CreateBr(HeaderBB);
      OffPhi->addIncoming(NextOff, BodyBB);

      // Exit from preload.
      IRBuilder<> BExit(ExitBB);
      createBarrier(BExit, *M);
      BExit.CreateBr(ContBB);

      // Compute block: load from shared tile, stage output in OutTile.
      IRBuilder<> BCont(&*ContBB->getFirstInsertionPt());

      Value *SharedIdx = BCont.CreateMul(
          Lid0,
          ConstantInt::get(BCont.getInt64Ty(), Match.StrideElems),
          "shared.idx");

      Value *SharedInBounds =
          BCont.CreateICmpULT(SharedIdx, ValidTileElems, "shared.in.bounds");

      Value *SafeIdx = BCont.CreateSelect(
          SharedInBounds,
          SharedIdx,
          ConstantInt::get(BCont.getInt64Ty(), 0),
          "shared.safe.idx");

      Value *SharedPtr =
          BCont.CreateGEP(Match.ElemTy, Tile, SafeIdx, "shared.ptr");

      LoadInst *SharedLoad =
          BCont.CreateLoad(Match.ElemTy, SharedPtr, "shared.load");

      // Replace original strided global load with shared load.
      LI->replaceAllUsesWith(SharedLoad);
      LI->eraseFromParent();

      // IRBuilder<> BContAfter(ContBB);
      IRBuilder<> BContAfter(ContBB->getTerminator());

      Value *OutIdx = Lid0;
      Value *OutTilePtr = BContAfter.CreateGEP(
          Match.ElemTy,
          OutTile,
          OutIdx,
          "tile.out.ptr");
      BContAfter.CreateStore(SharedLoad, OutTilePtr);

      createBarrier(BContAfter, *M);

      outs() << "[TileRemapPass] remapped strided load in function="
             << F.getName()
             << " stride_elems=" << Match.StrideElems
             << " const_elems=" << Match.ConstElems
             << "\n";

      Changed = true;
      verifyFunction(F, &errs());
    }

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

struct CoalescingPass : public PassInfoMixin<CoalescingPass>
{
  // PreservedAnalyses run(Function &F, FunctionAnalysisManager &)
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM)
  {

    errs() << "ENTER run: " << F.getName() << "\n";

    if (shouldSkipFunction(F))
    {
      errs() << "SKIP function: " << F.getName() << "\n";
      return PreservedAnalyses::all();
    }

    auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
    auto &LI = AM.getResult<LoopAnalysis>(F);
    auto &DT = AM.getResult<DominatorTreeAnalysis>(F);

    (void)LI;
    (void)DT;

    const DataLayout &DL = F.getParent()->getDataLayout();
    std::vector<AccessInfo> Accesses;

    Value *TidXVal = findTidXValue(F);
    Value *TidYVal = findTidYValue(F);

    const SCEV *TidXS = TidXVal ? SE.getSCEV(TidXVal) : nullptr;
    const SCEV *TidYS = TidYVal ? SE.getSCEV(TidYVal) : nullptr;

    if (TidXS)
      outs() << "[SCEV] tid_x=" << *TidXS << "\n";
    if (TidYS)
      outs() << "[SCEV] tid_y=" << *TidYS << "\n";

    for (BasicBlock &BB : F)
    {
      for (Instruction &I : BB)
      {
        Value *Ptr = nullptr;
        GetElementPtrInst *GEP = nullptr;
        uint64_t AccessSizeBytes = 0;
        std::string Kind;

        if (auto *LI = dyn_cast<LoadInst>(&I))
        {
          Ptr = LI->getPointerOperand();
          GEP = dyn_cast<GetElementPtrInst>(stripSimpleCasts(Ptr));
          if (!GEP)
            continue;

          AccessSizeBytes = getTypeSizeInBytes(DL, LI->getType());
          if (AccessSizeBytes == 0)
            continue;

          Kind = "load";
        }
        else if (auto *SI = dyn_cast<StoreInst>(&I))
        {
          Ptr = SI->getPointerOperand();
          GEP = dyn_cast<GetElementPtrInst>(stripSimpleCasts(Ptr));
          if (!GEP)
            continue;

          AccessSizeBytes =
              getTypeSizeInBytes(DL, SI->getValueOperand()->getType());
          if (AccessSizeBytes == 0)
            continue;

          Kind = "store";
        }
        else
        {
          continue;
        }

        // Optional debugging
        outs() << "    RAW PTR: ";
        Ptr->printAsOperand(outs(), false);
        outs() << "\n";

        Value *Stripped = stripSimpleCasts(Ptr);
        outs() << "    STRIPPED PTR: ";
        Stripped->printAsOperand(outs(), false);
        outs() << "\n";

        const SCEV *PtrS = SE.getSCEV(Ptr);
        outs() << "    PTR SCEV: " << *PtrS << "\n";

        const SCEV *PtrBaseS = SE.getPointerBase(PtrS);
        outs() << "    PTR BASE SCEV: " << *PtrBaseS << "\n";

        if (GEP)
        {
          outs() << "    GEP PTR OPERAND: ";
          GEP->getPointerOperand()->printAsOperand(outs(), false);
          outs() << "\n";

          outs() << "    GEP INDICES:\n";
          unsigned IdxNo = 0;
          for (Value *Idx : GEP->indices())
          {

            outs() << "      idx" << IdxNo++ << "=";
            Idx->printAsOperand(outs(), false);
            outs() << "\n";

            dumpSCEVInfoForIndex(SE, TidXS, TidYS, Idx);
            outs().flush();
            // outs() << "      idx" << IdxNo++ << "=";
            // Idx->printAsOperand(outs(), false);

            // // AffineExpr IdxExpr = parseAffine(Idx);
            // outs() << " affine=" << formatAffineExpr(IdxExpr);

            // ThreadVarKind TV = getThreadVarKind(Idx);
            // if (TV == ThreadVarKind::TidX)
            //   outs() << " kind=tid_x";
            // else if (TV == ThreadVarKind::TidY)
            //   outs() << " kind=tid_y";
            // else
            //   outs() << " kind=" << affineThreadKindName(IdxExpr);

            // int64_t C = 0;
            // if (getConstInt(Idx, C))
            //   outs() << " const=" << C;

            // outs() << "\n";

            // dumpIndexDef(Idx);
          }
        }

        // AffineExpr ByteExpr = buildByteOffsetExprRecursive(Ptr, DL);
        // WarpAccessInfo WI = analyzeWarp(ByteExpr, AccessSizeBytes);
        // std::string ClassName = classifyAccess(ByteExpr, AccessSizeBytes, WI);

        // AccessInfo AI;
        // AI.Kind = Kind;
        // AI.BaseValueStr = valueToString(Ptr);
        // AI.BaseName = getBaseName(Ptr);
        // // AI.OffsetStr = formatAffineExpr(ByteExpr);
        // // AI.ThreadDependent = ByteExpr.dependsOnThreads();
        // // AI.StrideXBytes = computeStrideXBytes(ByteExpr);
        // AI.AccessSize = AccessSizeBytes;
        // AI.Exact = WI.Exact;
        // AI.Contiguous = WI.Contiguous;
        // AI.Monotonic = WI.Monotonic;
        // AI.Broadcast = WI.Broadcast;
        // AI.ClassName = ClassName;
        // AI.Suggestion = suggestOptimization(AI);

        // AI.Action = chooseTransform(AI);
        // AI.ActionReason = explainTransformChoice(AI);

        // Accesses.push_back(std::move(AI));
      }
    }

    outs() << "[CoalescingPass] function=" << F.getName() << "\n";

    unsigned KeepCount = 0, BroadcastCount = 0, TileCount = 0, SkipCount = 0;

    for (const auto &AI : Accesses)
    {
      outs() << "  access=" << AI.Kind
             << " base=" << AI.BaseName
             << " byte_offset=" << AI.OffsetStr
             << " thread_dependent=" << (AI.ThreadDependent ? "yes" : "no")
             << " stride_x_bytes=" << AI.StrideXBytes
             << " access_size=" << AI.AccessSize
             << " exact=" << (AI.Exact ? "yes" : "no")
             << " contiguous=" << (AI.Contiguous ? "yes" : "no")
             << " monotonic=" << (AI.Monotonic ? "yes" : "no")
             << " broadcast=" << (AI.Broadcast ? "yes" : "no")
             << " class=" << AI.ClassName
             << "\n";

      outs() << "    " << AI.Suggestion << "\n";
      outs() << "    action=" << transformKindName(AI.Action) << "\n";
      outs() << "    reason=" << AI.ActionReason << "\n";

      switch (AI.Action)
      {
      case TransformKind::KeepGlobal:
        ++KeepCount;
        break;
      case TransformKind::BroadcastReuse:
        ++BroadcastCount;
        break;
      case TransformKind::TileRemap:
        ++TileCount;
        break;
      case TransformKind::SkipUnknown:
        ++SkipCount;
        break;
      }
    }

    outs() << "  summary:"
           << " keep=" << KeepCount
           << " broadcast_reuse=" << BroadcastCount
           << " tile_remap=" << TileCount
           << " skip=" << SkipCount
           << "\n";

    return PreservedAnalyses::all();
  }
};

struct CoalescingRewritePass : public PassInfoMixin<CoalescingRewritePass>
{
  // PreservedAnalyses run(Function &F, FunctionAnalysisManager &)
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM)
  {
    if (shouldSkipFunction(F))
      return PreservedAnalyses::all();

    const DataLayout &DL = F.getParent()->getDataLayout();
    bool Changed = false;

    errs() << "ENTER rewrite run: " << F.getName() << "\n";

    for (BasicBlock &BB : F)
    {
      DenseMap<std::pair<Value *, Type *>, LoadInst *> AvailableInvariantLoads;
      SmallVector<Instruction *, 8> ToErase;

      for (Instruction &I : BB)
      {
        auto *LI = dyn_cast<LoadInst>(&I);
        if (!LI)
          continue;

        if (LI->isVolatile() || LI->isAtomic())
          continue;

        std::string ClassName = classifyLoadForRewrite(LI, DL);

        // Only rewrite the easy/safe subset for now.
        bool Reusable =
            (ClassName == "INVARIANT_ACROSS_X" ||
             ClassName == "THREAD_INVARIANT");

        if (!Reusable)
        {
          // For now, do not rewrite STRIDED / TILE_REMAP cases.
          continue;
        }

        Value *Ptr = LI->getPointerOperand();
        auto K = std::make_pair(Ptr, LI->getType());

        auto It = AvailableInvariantLoads.find(K);
        if (It != AvailableInvariantLoads.end())
        {
          LoadInst *Prev = It->second;

          // Be conservative: only fold exact same type and pointer.
          if (Prev && Prev->getParent() == &BB &&
              Prev->getType() == LI->getType())
          {
            outs() << "[CoalescingRewritePass] reuse invariant load in function="
                   << F.getName()
                   << " bb=";
            BB.printAsOperand(outs(), false);
            outs() << " ptr=";
            Ptr->printAsOperand(outs(), false);
            outs() << " class=" << ClassName << "\n";

            LI->replaceAllUsesWith(Prev);
            ToErase.push_back(LI);
            Changed = true;
            continue;
          }
        }

        AvailableInvariantLoads[K] = LI;
      }

      for (Instruction *Dead : ToErase)
        Dead->eraseFromParent();
    }

    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo
llvmGetPassPluginInfo()
{
  errs() << "PLUGIN ENTRY LOADED\n";
  return {
      LLVM_PLUGIN_API_VERSION,
      "CoalescingPass",
      LLVM_VERSION_STRING,
      [](PassBuilder &PB)
      {
        errs() << "REGISTERING CALLBACK\n";
        PB.registerPipelineParsingCallback(
            [](StringRef Name, FunctionPassManager &FPM,
               ArrayRef<PassBuilder::PipelineElement>)
            {
              errs() << "PIPELINE NAME: " << Name << "\n";

              if (Name == "coalescing-pass")
              {
                errs() << "ADDING CoalescingPass\n";
                FPM.addPass(CoalescingPass());
                return true;
              }

              if (Name == "coalescing-rewrite-pass")
              {
                errs() << "ADDING CoalescingRewritePass\n";
                FPM.addPass(CoalescingRewritePass());
                return true;
              }

              if (Name == "tile-remap-pass")
              {
                errs() << "ADDING TileRemapPass\n";
                FPM.addPass(TileRemapPass());
                return true;
              }

              if (Name == "coalescing-pipeline")
              {
                errs() << "ADDING CoalescingPass + CoalescingRewritePass\n";
                FPM.addPass(CoalescingPass());
                FPM.addPass(CoalescingRewritePass());
                return true;
              }

              if (Name == "coalescing-full-pipeline")
              {
                errs() << "ADDING CoalescingPass + CoalescingRewritePass + TileRemapPass\n";
                FPM.addPass(CoalescingPass());
                FPM.addPass(CoalescingRewritePass());
                FPM.addPass(TileRemapPass());
                return true;
              }

              return false;
            });
      }};
}