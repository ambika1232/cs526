#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"


#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
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


#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>

using namespace llvm;

namespace {

enum class ThreadVarKind {
  None,
  TidX,
  TidY
};

struct AffineExpr {
  bool Known = true;
  bool HasUnknownInvariant = false;

  int64_t CoeffTidX = 0;
  int64_t CoeffTidY = 0;
  int64_t Constant = 0;

  bool dependsOnThreads() const {
    return CoeffTidX != 0 || CoeffTidY != 0;
  }
};

static AffineExpr invalidExpr() {
  AffineExpr E;
  E.Known = false;
  return E;
}

static AffineExpr makeConst(int64_t C) {
  AffineExpr E;
  E.Known = true;
  E.Constant = C;
  return E;
}

static AffineExpr makeUnknownInvariant() {
  AffineExpr E;
  E.Known = true;
  E.HasUnknownInvariant = true;
  return E;
}

static AffineExpr makeTidX() {
  AffineExpr E;
  E.Known = true;
  E.CoeffTidX = 1;
  return E;
}

static AffineExpr makeTidY() {
  AffineExpr E;
  E.Known = true;
  E.CoeffTidY = 1;
  return E;
}

static Value *stripSimpleCasts(Value *V) {
  while (true) {
    if (auto *CI = dyn_cast<CastInst>(V)) {
      V = CI->getOperand(0);
      continue;
    }
    if (auto *CE = dyn_cast<ConstantExpr>(V)) {
      if (CE->isCast()) {
        V = CE->getOperand(0);
        continue;
      }
    }
    break;
  }
  return V;
}

static bool getConstInt(Value *V, int64_t &C) {
  V = stripSimpleCasts(V);
  if (auto *CI = dyn_cast<ConstantInt>(V)) {
    C = CI->getSExtValue();
    return true;
  }
  return false;
}

static ThreadVarKind getThreadVarKind(Value *V) {
  V = stripSimpleCasts(V);

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

static std::string formatAffineExpr(const AffineExpr &E) {
  if (!E.Known)
    return "unknown";

  std::string S;
  bool First = true;

  auto appendTerm = [&](int64_t Coeff, StringRef Name) {
    if (Coeff == 0)
      return;

    std::string Term;
    if (Coeff == 1)
      Term = Name.str();
    else if (Coeff == -1)
      Term = "-" + Name.str();
    else
      Term = std::to_string(Coeff) + "*" + Name.str();

    if (First) {
      S += Term;
      First = false;
    } else {
      if (Coeff > 0)
        S += "+" + Term;
      else
        S += Term;
    }
  };

  appendTerm(E.CoeffTidX, "tid_x");
  appendTerm(E.CoeffTidY, "tid_y");

  if (E.Constant != 0) {
    if (First) {
      S += std::to_string(E.Constant);
      First = false;
    } else if (E.Constant > 0) {
      S += "+" + std::to_string(E.Constant);
    } else {
      S += std::to_string(E.Constant);
    }
  } else if (First && !E.HasUnknownInvariant) {
    S += "0";
    First = false;
  }

  if (E.HasUnknownInvariant) {
    if (First)
      S += "sym";
    else
      S += "+sym";
  }

  return S;
}

static AffineExpr combineAdd(const AffineExpr &A, const AffineExpr &B) {
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

static AffineExpr scaleExpr(const AffineExpr &A, int64_t Scale) {
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

static AffineExpr divideExprIfPossible(const AffineExpr &A, int64_t Div) {
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

static bool isLoopPHI(Value *V) {
  auto *PHI = dyn_cast<PHINode>(V);
  if (!PHI)
    return false;

  for (Value *Incoming : PHI->incoming_values()) {
    Incoming = stripSimpleCasts(Incoming);

    if (Incoming == PHI)
      continue;

    if (isa<ConstantInt>(Incoming))
      continue;

    if (auto *BO = dyn_cast<BinaryOperator>(Incoming)) {
      if (BO->getOpcode() == Instruction::Add ||
          BO->getOpcode() == Instruction::Sub) {
        bool UsesSelf = false;
        bool HasConst = false;

        for (Value *Op : BO->operands()) {
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

static bool isInvariantButUnknown(Value *V) {
  V = stripSimpleCasts(V);

  if (isa<ConstantInt>(V))
    return true;

  switch (getThreadVarKind(V)) {
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

  if (auto *BO = dyn_cast<BinaryOperator>(V)) {
    return isInvariantButUnknown(BO->getOperand(0)) &&
           isInvariantButUnknown(BO->getOperand(1));
  }

  if (auto *PHI = dyn_cast<PHINode>(V)) {
    (void)PHI;
    return isLoopPHI(V);
  }

  if (auto *SI = dyn_cast<SelectInst>(V)) {
    return isInvariantButUnknown(SI->getTrueValue()) &&
           isInvariantButUnknown(SI->getFalseValue());
  }

  if (auto *CB = dyn_cast<CallBase>(V)) {
    (void)CB;
    return false;
  }

  return false;
}

static AffineExpr parseAffine(Value *V) {
  V = stripSimpleCasts(V);

  switch (getThreadVarKind(V)) {
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

  if (auto *SI = dyn_cast<SelectInst>(V)) {
    AffineExpr T = parseAffine(SI->getTrueValue());
    AffineExpr F = parseAffine(SI->getFalseValue());

    if (T.Known && F.Known &&
        T.CoeffTidX == F.CoeffTidX &&
        T.CoeffTidY == F.CoeffTidY) {
      AffineExpr E;
      E.Known = true;
      E.CoeffTidX = T.CoeffTidX;
      E.CoeffTidY = T.CoeffTidY;
      E.HasUnknownInvariant = true;
      return E;
    }

    if (isInvariantButUnknown(V))
      return makeUnknownInvariant();

    return invalidExpr();
  }

  auto *BO = dyn_cast<BinaryOperator>(V);
  if (!BO) {
    if (isInvariantButUnknown(V))
      return makeUnknownInvariant();
    return invalidExpr();
  }

  switch (BO->getOpcode()) {
  case Instruction::Add: {
    AffineExpr L = parseAffine(BO->getOperand(0));
    AffineExpr R = parseAffine(BO->getOperand(1));
    return combineAdd(L, R);
  }

  case Instruction::Sub: {
    AffineExpr L = parseAffine(BO->getOperand(0));
    AffineExpr R = parseAffine(BO->getOperand(1));
    if (!L.Known || !R.Known)
      return invalidExpr();
    return combineAdd(L, scaleExpr(R, -1));
  }

  case Instruction::Mul: {
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
        RopInvariantUnknown) {
      return makeUnknownInvariant();
    }

    if (R.Known && !R.HasUnknownInvariant && R.CoeffTidX == 0 &&
        LopInvariantUnknown) {
      return makeUnknownInvariant();
    }

    if (LopInvariantUnknown && RopInvariantUnknown)
      return makeUnknownInvariant();

    return invalidExpr();
  }

  case Instruction::Shl: {
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
  case Instruction::LShr: {
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

static uint64_t getTypeSizeInBytes(const DataLayout &DL, Type *Ty) {
  if (!Ty || !Ty->isSized())
    return 0;
  return DL.getTypeAllocSize(Ty);
}

static AffineExpr buildByteOffsetExpr(GetElementPtrInst *GEP,
                                      const DataLayout &DL) {
  AffineExpr Total = makeConst(0);

  for (auto GTI = llvm::gep_type_begin(GEP), GTE = llvm::gep_type_end(GEP);
       GTI != GTE; ++GTI) {
    Value *Idx = GTI.getOperand();

    if (StructType *STy = GTI.getStructTypeOrNull()) {
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

static bool canEvaluateExactly(const AffineExpr &E) {
  return E.Known && !E.HasUnknownInvariant;
}

static int64_t evaluateAffineBytes(const AffineExpr &E, int64_t TidX,
                                   int64_t TidY) {
  return E.CoeffTidX * TidX + E.CoeffTidY * TidY + E.Constant;
}

static int64_t computeStrideXBytes(const AffineExpr &E) {
  if (!E.Known)
    return 0;
  return std::llabs(E.CoeffTidX);
}

struct WarpAccessInfo {
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
                                  uint64_t AccessSizeBytes) {
  WarpAccessInfo Info;
  if (!ByteExpr.Known || AccessSizeBytes == 0)
    return Info;

  Info.Valid = true;
  Info.Exact = canEvaluateExactly(ByteExpr);

  if (!Info.Exact) {
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
                  [&](int64_t A) { return A == Addrs.front(); });

  Info.Monotonic = true;
  for (size_t I = 1; I < Addrs.size(); ++I) {
    if (Addrs[I] < Addrs[I - 1]) {
      Info.Monotonic = false;
      break;
    }
  }

  Info.Contiguous = true;
  for (size_t I = 1; I < Addrs.size(); ++I) {
    if (Addrs[I] != Addrs[I - 1] + static_cast<int64_t>(AccessSizeBytes)) {
      Info.Contiguous = false;
      break;
    }
  }

  Info.Aligned128 = ((Addrs.front() % 128) == 0);

  std::set<int64_t> Segments;
  for (int Lane = 0; Lane < 32; ++Lane) {
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
                                  const WarpAccessInfo &WI) {
  if (!ByteExpr.Known) {
    return "UNKNOWN";
  }

  if (!ByteExpr.dependsOnThreads()) {
    if (ByteExpr.HasUnknownInvariant) {
      return "INVARIANT_ACROSS_X";
    }
    return "THREAD_INVARIANT";
  }

  int64_t StrideBytes = computeStrideXBytes(ByteExpr);

  if (StrideBytes == 0) {
    return "INVARIANT_ACROSS_X";
  }

  if (StrideBytes == static_cast<int64_t>(AccessSizeBytes)) {
    if (WI.Exact) {
      if (WI.Num128BTransactions == 1 && WI.Aligned128) {
        return "FULLY_COALESCED";
      }
      if (WI.Num128BTransactions <= 2) {
        return "COALESCED_BUT_MISALIGNED";
      }
      return "PARTIALLY_COALESCED";
    }
    return "LIKELY_COALESCED";
  }

  if (StrideBytes > static_cast<int64_t>(AccessSizeBytes)) {
    return "STRIDED";
  }

  if (StrideBytes < static_cast<int64_t>(AccessSizeBytes)) {
    return "OVERLAPPED_OR_PACKED";
  }

  return "UNKNOWN";
}

static void printAccessSummary(StringRef AccessKind, GetElementPtrInst *GEP,
                               const AffineExpr &ByteExpr,
                               uint64_t AccessSizeBytes) {
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

  if (WI.Valid && WI.Exact) {
    outs() << " tx128=" << WI.Num128BTransactions;
    outs() << " aligned128=" << (WI.Aligned128 ? "yes" : "no");
    outs() << " contiguous=" << (WI.Contiguous ? "yes" : "no");
    outs() << " monotonic=" << (WI.Monotonic ? "yes" : "no");
    outs() << " broadcast=" << (WI.Broadcast ? "yes" : "no");
  } else if (WI.Valid) {
    outs() << " contiguous=" << (WI.Contiguous ? "yes" : "no");
    outs() << " monotonic=" << (WI.Monotonic ? "yes" : "no");
    outs() << " broadcast=" << (WI.Broadcast ? "yes" : "no");
  }

  outs() << " class="
         << classifyAccess(ByteExpr, AccessSizeBytes, WI);
  outs() << "\n";
}

struct CoalescingPass : public PassInfoMixin<CoalescingPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    const DataLayout &DL = F.getParent()->getDataLayout();

    outs() << "[CoalescingPass] function=" << F.getName() << "\n";

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          Value *Ptr = LI->getPointerOperand();
          auto *GEP = dyn_cast<GetElementPtrInst>(stripSimpleCasts(Ptr));
          if (!GEP)
            continue;

          uint64_t AccessSizeBytes = getTypeSizeInBytes(DL, LI->getType());
          if (AccessSizeBytes == 0)
            continue;

          AffineExpr ByteExpr = buildByteOffsetExpr(GEP, DL);
          printAccessSummary("load", GEP, ByteExpr, AccessSizeBytes);

        } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
          Value *Ptr = SI->getPointerOperand();
          auto *GEP = dyn_cast<GetElementPtrInst>(stripSimpleCasts(Ptr));
          if (!GEP)
            continue;

          uint64_t AccessSizeBytes =
              getTypeSizeInBytes(DL, SI->getValueOperand()->getType());
          if (AccessSizeBytes == 0)
            continue;

          AffineExpr ByteExpr = buildByteOffsetExpr(GEP, DL);
          printAccessSummary("store", GEP, ByteExpr, AccessSizeBytes);
        }
      }
    }

    return PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {
      LLVM_PLUGIN_API_VERSION,
      "CoalescingPass",
      LLVM_VERSION_STRING,
      [](PassBuilder &PB) {
        PB.registerPipelineParsingCallback(
            [](StringRef Name, FunctionPassManager &FPM,
               ArrayRef<PassBuilder::PipelineElement>) {
              if (Name == "coalescing-pass") {
                FPM.addPass(CoalescingPass());
                return true;
              }
              return false;
            });
      }};
}