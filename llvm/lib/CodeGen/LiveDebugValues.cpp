//===- LiveDebugValues.cpp - Tracking Debug Value MIs ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This pass implements a data flow analysis that propagates debug location
/// information by inserting additional DBG_VALUE insts into the machine
/// instruction stream. Before running, each DBG_VALUE inst corresponds to a
/// source assignment of a variable. Afterwards, a DBG_VALUE inst specifies a
/// variable location for the current basic block (see SourceLevelDebugging.rst).
///
/// This is a separate pass from DbgValueHistoryCalculator to facilitate
/// testing and improve modularity.
///
/// Each variable location is represented by a VarLoc object that identifies the
/// source variable, its current machine-location, and the DBG_VALUE inst that
/// specifies the location. Each VarLoc is indexed in the (function-scope)
/// VarLocMap, giving each VarLoc a unique index. Rather than operate directly
/// on machine locations, the dataflow analysis in this pass identifies
/// locations by their index in the VarLocMap, meaning all the variable
/// locations in a block can be described by a sparse vector of VarLocMap
/// indexes.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/CoalescingBitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/CodeGen/LexicalScopes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "livedebugvalues"

STATISTIC(NumInserted, "Number of DBG_VALUE instructions inserted");
STATISTIC(NumRemoved, "Number of DBG_VALUE instructions removed");

static cl::opt<bool> EmulateOldLDV(
    "word-wrap-like-word97", cl::Hidden,
    cl::desc("Act like old LiveDebugValues did"),
    cl::init(true));

namespace {

// The location at which a spilled variable resides. It consists of a
// register and an offset.
struct SpillLoc {
  unsigned SpillBase;
  int SpillOffset;
  bool operator==(const SpillLoc &Other) const {
    return SpillBase == Other.SpillBase && SpillOffset == Other.SpillOffset;
  }
  bool operator<(const SpillLoc &Other) const {
    return std::tie(SpillBase, SpillOffset) < std::tie(Other.SpillBase, Other.SpillOffset);
  }
};

// This is purely a number that's slightly more strongly typed.
enum LocIdx { limin = 0, limax = UINT_MAX };

class ValueIDNum {
public:
  uint64_t BlockNo : 16;
  uint64_t InstNo : 20;
  LocIdx LocNo : 14; // No idea why this works, it shouldn't!

  uint64_t asU64() const {
    uint64_t tmp_block = BlockNo;
    uint64_t tmp_inst = InstNo;
    return tmp_block << 34ull | tmp_inst << 14 | LocNo;
  }

  static ValueIDNum fromU64(uint64_t v) {
    LocIdx l = LocIdx(v & 0x3FFF);
    return {v >> 34ull, ((v >> 14) & 0xFFFFF), l};
  }

 bool operator<(const ValueIDNum &Other) const {
   return asU64() < Other.asU64();
 }

 bool operator==(const ValueIDNum &Other) const {
   return std::tie(BlockNo, InstNo, LocNo) ==
          std::tie(Other.BlockNo, Other.InstNo, Other.LocNo);
 }

   bool operator!=(const ValueIDNum &Other) const {
    return !(*this == Other);
   }

  std::string asString(const std::string &mlocname) const {
    return Twine("bb ").concat(
           Twine(BlockNo).concat(
           Twine(" inst ").concat(
           Twine(InstNo).concat(
           Twine(" loc ").concat(
           Twine(mlocname)))))).str();
  }
};

} // end anon namespace

namespace llvm {
template <> struct DenseMapInfo<ValueIDNum> {
  // NB, there's a risk of overlap of uint64_max with legitmate numbering if
  // there are very many machine locations. Fix by not bit packing so hard.
  static const uint64_t MaxVal = std::numeric_limits<uint64_t>::max();

  static inline ValueIDNum getEmptyKey() { return ValueIDNum::fromU64(MaxVal); }

  static inline ValueIDNum getTombstoneKey() { return ValueIDNum::fromU64(MaxVal - 1); }

  static unsigned getHashValue(ValueIDNum num) {
    return hash_value(num.asU64());
  }

  static bool isEqual(const ValueIDNum &A, const ValueIDNum &B) { return A == B; }
};

// Misery
template <> struct DenseMapInfo<LocIdx> {
  static const int MaxVal = std::numeric_limits<int>::max();

  static inline LocIdx getEmptyKey() { return LocIdx(MaxVal); }

  static inline LocIdx getTombstoneKey() { return LocIdx(MaxVal-1); }

  static unsigned getHashValue(LocIdx Num) {
    return hash_value((unsigned)Num);
  }

  static bool isEqual(LocIdx A, LocIdx B) { return A == B; }
};



} // end namespace llvm


namespace {

class VarLocPos {
public:
  ValueIDNum ID;
  LocIdx CurrentLoc : 14;

  uint64_t asU64() const {
    return ID.asU64() << 14 | CurrentLoc;
  }

  static VarLocPos fromU64(uint64_t v) {
    return {ValueIDNum::fromU64(v >> 14), LocIdx(v & 0x3FFF)};
  }

  bool operator==(const VarLocPos &Other) const {
    return std::tie(ID, CurrentLoc) == std::tie(Other.ID, Other.CurrentLoc);
  }

  std::string asString(const std::string &curname, const std::string &defname) const {
    return Twine("VLP(").concat(ID.asString(defname)).concat(",cur ").concat(curname).concat(")").str();
  }
};

typedef std::pair<const DIExpression *, bool> MetaVal;

class MLocTracker {
public:
  MachineFunction &MF;
  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;
  const TargetLowering &TLI;

  std::vector<LocIdx> LocIDToLocIdx;
  DenseMap<LocIdx, unsigned> LocIdxToLocID;
  std::vector<ValueIDNum> LocIdxToIDNum;
  UniqueVector<SpillLoc> SpillLocs;
  unsigned lolwat_cur_bb;
  unsigned NumRegs;

  SmallVector<std::pair<const MachineOperand *, unsigned>, 32> Masks;

  MLocTracker(MachineFunction &MF, const TargetInstrInfo &TII, const TargetRegisterInfo &TRI, const TargetLowering &TLI)
    : MF(MF), TII(TII), TRI(TRI), TLI(TLI) {
    NumRegs = TRI.getNumRegs();
    reset();
    LocIdxToIDNum.push_back({0, 0, LocIdx(0)});
    LocIDToLocIdx.resize(NumRegs);
    memset(&LocIDToLocIdx[0], 0, NumRegs * sizeof(LocIdx));
    LocIDToLocIdx[0] = LocIdx(0);
    LocIdxToLocID[LocIdx(0)] = 0;
  }

  unsigned getLocID(unsigned RegOrSpill, bool isSpill) {
    return (isSpill) ? RegOrSpill + NumRegs : RegOrSpill;
  }

  VarLocPos getVarLocPos(LocIdx Idx) const {
    assert(Idx < LocIdxToIDNum.size());
    return {LocIdxToIDNum[Idx], Idx};
  }

  unsigned getNumLocs(void) const {
    return LocIdxToIDNum.size();
  }

  void setMPhis(unsigned cur_bb) {
    lolwat_cur_bb = cur_bb;
    for (unsigned ID = 1; ID < LocIdxToIDNum.size(); ++ID) {
      LocIdxToIDNum[LocIdx(ID)] = {cur_bb, 0, LocIdx(ID)};
    }
  }

  void loadFromArray(uint64_t *Locs, unsigned cur_bb) {
    lolwat_cur_bb = cur_bb;
    // Quickly reset everything to being itself at inst 0, representing a phi.
    for (unsigned ID = 1; ID < LocIdxToIDNum.size(); ++ID) {
      LocIdxToIDNum[LocIdx(ID)] = ValueIDNum::fromU64(Locs[ID]);
    }
  }

  void reset(void) {
    memset(&LocIdxToIDNum[0], 0, LocIdxToIDNum.size() * sizeof(ValueIDNum));
    Masks.clear();
  }

  void clear(void) {
    reset();
    LocIDToLocIdx.clear();
    LocIdxToLocID.clear();
    LocIdxToIDNum.clear();
    //SpillsToMLocs.reset(); XXX can't reset?
    SpillLocs = decltype(SpillLocs)();

    LocIDToLocIdx.resize(NumRegs);
    memset(&LocIDToLocIdx[0], 0, NumRegs * sizeof(LocIdx));
  }

  void setMLoc(LocIdx L, ValueIDNum Num) {
    assert(L < LocIdxToIDNum.size());
    LocIdxToIDNum[L] = Num;
  }

  void bumpRegister(unsigned ID, LocIdx &Ref) {
     assert(ID != 0);
    if (Ref == 0) {
      LocIdx NewIdx = LocIdx(LocIdxToIDNum.size());
      Ref = NewIdx;

      // Default: it's an mphi.
      ValueIDNum ValNum = {lolwat_cur_bb, 0, NewIdx};
      // Was this reg ever touched by a regmask?
      for (auto rit = Masks.rbegin(); rit != Masks.rend(); ++rit) {
        if (rit->first->clobbersPhysReg(ID))  {
          // There was an earlier def we skipped
          ValNum = {lolwat_cur_bb, rit->second, NewIdx};
          break;
        }
      }

      LocIdxToIDNum.push_back(ValNum);
      LocIdxToLocID[NewIdx] = ID;
    }
  }

  void defReg(Register r, unsigned bb, unsigned inst) {
    unsigned ID = getLocID(r, false);
    LocIdx &Idx = LocIDToLocIdx[ID];
    bumpRegister(ID, Idx);
    ValueIDNum id = {bb, inst, Idx};
    LocIdxToIDNum[Idx] = id;
  }

  void setReg(Register r, ValueIDNum id) {
    unsigned ID = getLocID(r, false);
    LocIdx &Idx = LocIDToLocIdx[ID];
    bumpRegister(ID, Idx);
    LocIdxToIDNum[Idx] = id;
  }

  ValueIDNum readReg(Register r) {
    unsigned ID = getLocID(r, false);
    LocIdx &Idx = LocIDToLocIdx[ID];
    bumpRegister(ID, Idx);
    return LocIdxToIDNum[Idx];
  }

  // Because we need to replicate values only having one location for now.
  void lolwipe(Register r) {
    unsigned ID = getLocID(r, false);
    LocIdx Idx = LocIDToLocIdx[ID];
    LocIdxToIDNum[Idx] = {0, 0, LocIdx(0)};
  }

  LocIdx getRegMLoc(Register r) {
    unsigned ID = getLocID(r, false);
    return LocIDToLocIdx[ID];
  }

  void writeRegMask(const MachineOperand *MO, unsigned cur_bb, unsigned inst_id) {
    // Def anything we already have that isn't preserved.
    unsigned SP = TLI.getStackPointerRegisterToSaveRestore();
    // Ensure SP exists, so that we don't override it later.
    unsigned ID = getLocID(SP, false);
    LocIdx &Idx = LocIDToLocIdx[ID];
    bumpRegister(ID, Idx);

    for (auto &P : LocIdxToLocID) {
      // Don't believe mask clobbering SP.
      if (P.second != 0 && P.second < NumRegs && P.second != SP &&
          MO->clobbersPhysReg(P.second))
        defReg(P.second, cur_bb, inst_id);
    }
    Masks.push_back(std::make_pair(MO, inst_id));
  }

  void setSpill(SpillLoc l, ValueIDNum id) {
    unsigned SpillID = SpillLocs.idFor(l);
    if (SpillID == 0) {
      SpillID = SpillLocs.insert(l);
      LocIDToLocIdx.push_back(LocIdx(0));
      unsigned L = getLocID(SpillID, true);
      LocIdx Idx = LocIdx(LocIdxToIDNum.size()); // New idx
      LocIDToLocIdx[L] = Idx;
      LocIdxToLocID[Idx] = L;
      LocIdxToIDNum.push_back(id);
    } else {
      unsigned L = getLocID(SpillID, true);
      LocIdx Idx = LocIDToLocIdx[L];
      LocIdxToIDNum[Idx] = id;
    }
  }

  ValueIDNum readSpill(SpillLoc l) {
    unsigned pos = SpillLocs.idFor(l);
    if (pos == 0)
      // Returning no location -> 0 means $noreg and some hand wavey position
      return {0, 0, LocIdx(0)};

    unsigned L = getLocID(pos, true);
    unsigned LocIdx = LocIDToLocIdx[L];
    return LocIdxToIDNum[LocIdx];
  }

  LocIdx getSpillMLoc(SpillLoc l) {
    unsigned SpillID = SpillLocs.idFor(l);
    if (SpillID == 0)
      return LocIdx(0);
    unsigned L = getLocID(SpillID, true);
    return LocIDToLocIdx[L];
  }

  bool isSpill(LocIdx Idx) const {
    auto it = LocIdxToLocID.find(Idx);
    assert(it != LocIdxToLocID.end());
    return it->second >= NumRegs;
  }

  std::string LocIdxToName(LocIdx Idx) const {
    auto it = LocIdxToLocID.find(Idx);
    assert(it != LocIdxToLocID.end());
    unsigned ID = it->second;
    if (ID >= NumRegs)
      return Twine("slot ").concat(Twine(ID - NumRegs)).str();
    else
      return TRI.getRegAsmName(ID).str();
  }

  std::string IDAsString(const ValueIDNum &num) const {
    std::string defname = LocIdxToName(num.LocNo);
    return num.asString(defname);
  }

  std::string PosAsString(const VarLocPos &Pos) const {
    std::string mlocname = LocIdxToName(Pos.CurrentLoc);
    std::string defname = LocIdxToName(Pos.ID.LocNo);
    return Pos.asString(mlocname, defname);
  }

  LLVM_DUMP_METHOD
  void dump() const {
    for (unsigned int ID = 0; ID < LocIdxToIDNum.size(); ++ID) {
      auto &num = LocIdxToIDNum[ID];
      if (num.LocNo == 0)
        continue;
      std::string mlocname = LocIdxToName(num.LocNo);
      std::string defname = num.asString(mlocname);
      dbgs() << LocIdxToName(LocIdx(ID)) << " --> " << defname << "\n";
    }
  }

  LLVM_DUMP_METHOD
  void dump_mloc_map() const {
    for (unsigned I = 0; I < LocIdxToIDNum.size(); ++I) {
      std::string foo = LocIdxToName(LocIdx(I));
      dbgs() << "Idx " << I << " " << foo << "\n";
    }
  }

  MachineInstrBuilder 
  emitLoc(LocIdx MLoc, const DebugVariable &Var, const MetaVal &meta) {
    DebugLoc DL = DebugLoc::get(0, 0, Var.getVariable()->getScope(), Var.getInlinedAt());
    auto MIB = BuildMI(MF, DL, TII.get(TargetOpcode::DBG_VALUE));

    const DIExpression *Expr = meta.first;
    unsigned Loc = LocIdxToLocID[MLoc];
    if (Loc >= NumRegs) {
      const SpillLoc &Spill = SpillLocs[Loc - NumRegs];
      Expr = DIExpression::prepend(Expr, DIExpression::ApplyOffset, Spill.SpillOffset);
      unsigned Base = Spill.SpillBase;
      MIB.addReg(Base, RegState::Debug);
      MIB.addImm(0);
   } else {
      MIB.addReg(Loc, RegState::Debug);
      if (meta.second)
        MIB.addImm(0);
      else
        MIB.addReg(0, RegState::Debug);
    }

    MIB.addMetadata(Var.getVariable());
    MIB.addMetadata(Expr);
    return MIB;
  }
};

class ValueRec {
public:
  ValueIDNum ID;
  Optional<MachineOperand> MO;
  MetaVal meta;

  typedef enum { Def, Const } KindT;
  KindT Kind;

  void dump(const MLocTracker *MTrack) const {
    if (Kind == Const) {
      MO->dump();
    } else {
      assert(Kind == Def);
      dbgs() << MTrack->IDAsString(ID);
    }
    if (meta.second)
      dbgs() << " indir";
    if (meta.first)
      dbgs() << " " << *meta.first;
  }

  bool operator==(const ValueRec &Other) const {
    if (Kind != Other.Kind)
      return false;
    if (Kind == Const && !MO->isIdenticalTo(*Other.MO))
      return false;
    else if (Kind == Def && ID != Other.ID)
      return false;

    return meta == Other.meta;
  }

  bool operator!=(const ValueRec &Other) const {
    return !(*this == Other);
  }
};

// Types for recording sets of variable fragments that overlap. For a given
// local variable, we record all other fragments of that variable that could
// overlap it, to reduce search time.
using FragmentOfVar =
    std::pair<const DILocalVariable *, DIExpression::FragmentInfo>;
using OverlapMap =
    DenseMap<FragmentOfVar, SmallVector<DIExpression::FragmentInfo, 1>>;

class VLocTracker {
public:
  // Map the DebugVariable to recent primary location ID.
  // Needs to be a mapvector because we determine order-in-the-input-MIR from
  // the order in this thing.
  MapVector<DebugVariable, ValueRec> Vars;
  MachineBasicBlock *MBB;

public:
  VLocTracker() {}

  void defVar(const MachineInstr &MI, ValueIDNum ID) {
    // XXX skipping overlapping fragments for now.
    assert(MI.isDebugValue());
    DebugVariable Var(MI.getDebugVariable(), MI.getDebugExpression(),
                      MI.getDebugLoc()->getInlinedAt());
    MetaVal m = {MI.getDebugExpression(), MI.getOperand(1).isImm()};
    Vars[Var] = {ID, None, m, ValueRec::Def};
  }

  void defVar(const MachineInstr &MI, const MachineOperand &MO) {
    // XXX skipping overlapping fragments for now.
    assert(MI.isDebugValue());
    DebugVariable Var(MI.getDebugVariable(), MI.getDebugExpression(),
                      MI.getDebugLoc()->getInlinedAt());
    MetaVal m = {MI.getDebugExpression(), MI.getOperand(1).isImm()};
    Vars[Var] = {{0, 0, LocIdx(0)}, MO, m, ValueRec::Const};
  }
};

class TransferTracker {
public:
  const TargetInstrInfo *TII;
  MLocTracker *mtracker;
  MachineFunction &MF;

  struct Transfer {
    MachineBasicBlock::iterator pos;
    MachineBasicBlock *MBB;
    std::vector<MachineInstr *> insts;
  };

  typedef std::pair<LocIdx, MetaVal> hahaloc;
  std::vector<Transfer> Transfers;
  std::vector<ValueIDNum> VarLocs;

  DenseMap<LocIdx, SmallSet<DebugVariable, 4>> ActiveMLocs;
  DenseMap<DebugVariable, hahaloc> ActiveVLocs;

  std::vector<MachineInstr *> PendingDbgValues;

  TransferTracker(const TargetInstrInfo *TII, MLocTracker *mtracker, MachineFunction &MF) : TII(TII), mtracker(mtracker), MF(MF) { }

  void loadInlocs(MachineBasicBlock &MBB, uint64_t *mlocs, SmallVectorImpl<std::pair<DebugVariable, ValueRec>> &vlocs, unsigned cur_bb, unsigned NumLocs) {  
    ActiveMLocs.clear();
    ActiveVLocs.clear();
    VarLocs.clear();
    VarLocs.resize(NumLocs);

    DenseMap<ValueIDNum, LocIdx> ValueToLoc;

    for (unsigned Idx = 1; Idx < NumLocs; ++Idx) {
      // Each mloc is a VarLocPos
      auto VNum = ValueIDNum::fromU64(mlocs[Idx]);
      VarLocs[Idx] = VNum;
      // Produce a map of value numbers to the current machine locs they live
      // in. There should only be one machine loc per value.
      //assert(ValueToLoc.find(VNum) == ValueToLoc.end()); // XXX expensie
      auto it = ValueToLoc.find(VNum);
      if (it == ValueToLoc.end() || mtracker->isSpill(it->second))
        ValueToLoc[VNum] = LocIdx(Idx);
    }

    // Now map variables to their current machine locs
    for (auto Var : vlocs) {
      if (Var.second.Kind == ValueRec::Const) {
        PendingDbgValues.push_back(emitMOLoc(*Var.second.MO, Var.first, Var.second.meta));
        continue;
      }

      // Value unavailable / has no machine loc -> define no location.
      auto hahait = ValueToLoc.find(Var.second.ID);
      if (hahait == ValueToLoc.end())
        continue;

      LocIdx m = hahait->second;
      ActiveVLocs[Var.first] = std::make_pair(m, Var.second.meta);
      ActiveMLocs[m].insert(Var.first);
      assert(m != 0);
      PendingDbgValues.push_back(mtracker->emitLoc(m, Var.first, Var.second.meta));
    }
    flushDbgValues(MBB.begin(), &MBB);
  }

  void flushDbgValues(MachineBasicBlock::iterator pos,
                      MachineBasicBlock *MBB) {
    if (PendingDbgValues.size() > 0) {
      Transfers.push_back({pos, MBB, PendingDbgValues});
      PendingDbgValues.clear();
    }
  }

  void redefVar(const MachineInstr &MI) {
    DebugVariable Var(MI.getDebugVariable(), MI.getDebugExpression(),
                      MI.getDebugLoc()->getInlinedAt());
    const MachineOperand &MO = MI.getOperand(0);

    // Erase any previous location,
    auto It = ActiveVLocs.find(Var);
    if (It != ActiveVLocs.end()) {
      ActiveMLocs[It->second.first].erase(Var);
    }

    // Insert a new vloc. Ignore non-register locations, we don't transfer
    // those, and can't current describe spill locs independently of regs.
    if (!MO.isReg() || MO.getReg() == 0) {
      if (It != ActiveVLocs.end())
        ActiveVLocs.erase(It);
      return;
    }

    Register Reg = MO.getReg();
    LocIdx MLoc = mtracker->getRegMLoc(Reg);
    MetaVal meta = {MI.getDebugExpression(), MI.getOperand(1).isImm()};

    // If that loc has been clobbered in the meantime, wipe its contents.
    if (mtracker->getVarLocPos(MLoc).ID != VarLocs[MLoc]) {
      for (auto &P : ActiveMLocs[MLoc]) {
        ActiveVLocs.erase(P);
      }
      ActiveMLocs[MLoc].clear();
      VarLocs[MLoc] = mtracker->getVarLocPos(MLoc).ID;
    }

    ActiveMLocs[MLoc].insert(Var);
    if (It == ActiveVLocs.end()) {
      ActiveVLocs.insert(std::make_pair(Var, std::make_pair(MLoc, meta)));
    } else {
      It->second.first = MLoc;
      It->second.second = meta;
    }
  }

  void clobberMloc(LocIdx mloc, MachineBasicBlock::iterator pos) {
    assert(mtracker->isSpill(mloc));
    auto It = ActiveMLocs.find(mloc);
    if (It == ActiveMLocs.end())
      return;

    VarLocs[mloc] = ValueIDNum{0, 0, LocIdx(0)};

    for (auto &Var : It->second) {
      auto ALoc = ActiveVLocs.find(Var);
      // Create an undef. We can't feed in a nullptr DIExpression alas,
      // so use the variables last expression.
      const DIExpression *Expr = ALoc->second.second.first;
      // XXX explicitly specify empty location?
      LocIdx Idx = LocIdx(0);
      PendingDbgValues.push_back(mtracker->emitLoc(Idx, Var, {Expr, false}));
      ActiveVLocs.erase(ALoc);
    }
    flushDbgValues(pos, nullptr);

    It->second.clear();
  }

  void transferMlocs(LocIdx src, LocIdx dst, MachineBasicBlock::iterator pos) {
    // Does src still contain the value num we expect? If not, it's been
    // clobbered in the meantime.
    if (VarLocs[src] != mtracker->getVarLocPos(src).ID)
      return;

    // Legitimate scenario on account of un-clobbered slot being assigned to?
    //assert(ActiveMLocs[dst].size() == 0);
    ActiveMLocs[dst] = ActiveMLocs[src];
    VarLocs[dst] = VarLocs[src];

    for (auto &Var : ActiveMLocs[src]) {
      auto it = ActiveVLocs.find(Var);
      assert(it != ActiveVLocs.end());
      it->second.first = dst;

      assert(dst != 0);
      MachineInstr *MI = mtracker->emitLoc(dst, Var, it->second.second);
      PendingDbgValues.push_back(MI);
    }
    ActiveMLocs[src].clear();
    flushDbgValues(pos, nullptr);

    // XXX XXX XXX "pretend to be old LDV".
    VarLocs[src] = ValueIDNum{0, 0, LocIdx(0)};
  }

  MachineInstrBuilder 
  emitMOLoc(const MachineOperand &MO,
              const DebugVariable &Var, const MetaVal &meta) {
    DebugLoc DL = DebugLoc::get(0, 0, Var.getVariable()->getScope(), Var.getInlinedAt());
    auto MIB = BuildMI(MF, DL, TII->get(TargetOpcode::DBG_VALUE));
    MIB.add(MO);
    if (meta.second)
      MIB.addImm(0);
    else
      MIB.addReg(0);
    MIB.addMetadata(Var.getVariable());
    MIB.addMetadata(meta.first);
    return MIB;
  }
};

class LiveDebugValues : public MachineFunctionPass {
private:
  using FragmentInfo = DIExpression::FragmentInfo;
  using OptFragmentInfo = Optional<DIExpression::FragmentInfo>;

  // Helper while building OverlapMap, a map of all fragments seen for a given
  // DILocalVariable.
  using VarToFragments =
      DenseMap<const DILocalVariable *, SmallSet<FragmentInfo, 4>>;

  typedef DenseMap<LocIdx, ValueIDNum> mloc_transfert;

  typedef DenseMap<const MachineBasicBlock *,
                   DenseMap<DebugVariable, ValueRec> *>
      LiveIdxT;

  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;
  const TargetFrameLowering *TFI;
  BitVector CalleeSavedRegs;
  LexicalScopes LS;

  MLocTracker *tracker;
  unsigned cur_bb;
  unsigned cur_inst;
  VLocTracker *vtracker;
  TransferTracker *ttracker;

  // Blocks which are artificial, i.e. blocks which exclusively contain
  // instructions without locations, or with line 0 locations.
  SmallPtrSet<const MachineBasicBlock *, 16> ArtificialBlocks;

  // Mapping of blocks to and from their RPOT order.
  DenseMap<unsigned int, MachineBasicBlock *> OrderToBB;
  DenseMap<MachineBasicBlock *, unsigned int> BBToOrder;

  OverlapMap OverlapFragments; // Map of overlapping variable fragments.
  VarToFragments SeenFragments;

  /// Tests whether this instruction is a spill to a stack location.
  bool isSpillInstruction(const MachineInstr &MI, MachineFunction *MF);

  /// Decide if @MI is a spill instruction and return true if it is. We use 2
  /// criteria to make this decision:
  /// - Is this instruction a store to a spill slot?
  /// - Is there a register operand that is both used and killed?
  /// TODO: Store optimization can fold spills into other stores (including
  /// other spills). We do not handle this yet (more than one memory operand).
  bool isLocationSpill(const MachineInstr &MI, MachineFunction *MF,
                       unsigned &Reg);

  /// If a given instruction is identified as a spill, return the spill location
  /// and set \p Reg to the spilled register.
  Optional<SpillLoc> isRestoreInstruction(const MachineInstr &MI,
                                                  MachineFunction *MF,
                                                  unsigned &Reg);
  /// Given a spill instruction, extract the register and offset used to
  /// address the spill location in a target independent way.
  SpillLoc extractSpillBaseRegAndOffset(const MachineInstr &MI);

  bool transferDebugValue(const MachineInstr &MI);
  bool transferSpillOrRestoreInst(MachineInstr &MI);
  bool transferRegisterCopy(MachineInstr &MI);
  void transferRegisterDef(MachineInstr &MI);

  void process(MachineInstr &MI);

  void accumulateFragmentMap(MachineInstr &MI);

  void produce_mloc_transfer_function(MachineFunction &MF,
                           std::vector<mloc_transfert> &MLocTransfer,
                           unsigned MaxNumBlocks);

  bool mloc_join(MachineBasicBlock &MBB,
                 SmallPtrSet<const MachineBasicBlock *, 16> &Visited,
                 uint64_t **OutLocs, uint64_t *InLocs);
  void mloc_dataflow(uint64_t **MInLocs, uint64_t **MOutLocs,
                     std::vector<mloc_transfert> &MLocTransfer);

  bool vloc_join_location(MachineBasicBlock &MBB, ValueRec &InLoc,
                          ValueRec &OLoc, uint64_t *InLocOutLocs,
                          uint64_t *OLOutlocs,
                          const LiveIdxT::mapped_type PrevInLocs, // is ptr
                          const DebugVariable &CurVar, bool ThisIsABackEdge);
  bool vloc_join(MachineBasicBlock &MBB, LiveIdxT &VLOCOutLocs,
                 LiveIdxT &VLOCInLocs,
                 SmallPtrSet<const MachineBasicBlock *, 16> *VLOCVisited,
                 unsigned cur_bb, const SmallSet<DebugVariable, 4> &AllVars,
                 uint64_t **MInLocs, uint64_t **MOutLocs,
                 SmallPtrSet<const MachineBasicBlock *, 8> &NonAssignBlocks);
  void vloc_dataflow(
      const LexicalScope *Scope,
      const SmallSet<DebugVariable, 4> &VarsWeCareAbout,
      SmallPtrSetImpl<MachineBasicBlock *> &AssignBlocks,
      SmallVectorImpl<SmallVector<std::pair<DebugVariable, ValueRec>, 8>>
          &Output,
      uint64_t **MOutLocs, uint64_t **MInLocs,
      SmallVectorImpl<VLocTracker> &AllTheVLocs);

  bool ExtendRanges(MachineFunction &MF);

public:
  static char ID;

  /// Default construct and initialize the pass.
  LiveDebugValues();

  /// Tell the pass manager which passes we depend on and what
  /// information we preserve.
  void getAnalysisUsage(AnalysisUsage &AU) const override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

  /// Calculate the liveness information for the given machine function.
  bool runOnMachineFunction(MachineFunction &MF) override;

  LLVM_DUMP_METHOD
  void dump_mloc_transfer(const mloc_transfert &mloc_transfer) const;
};

} // end anonymous namespace


//===----------------------------------------------------------------------===//
//            Implementation
//===----------------------------------------------------------------------===//

char LiveDebugValues::ID = 0;

char &llvm::LiveDebugValuesID = LiveDebugValues::ID;

INITIALIZE_PASS(LiveDebugValues, DEBUG_TYPE, "Live DEBUG_VALUE analysis",
                false, false)

/// Default construct and initialize the pass.
LiveDebugValues::LiveDebugValues() : MachineFunctionPass(ID) {
  initializeLiveDebugValuesPass(*PassRegistry::getPassRegistry());
}

/// Tell the pass manager which passes we depend on and what information we
/// preserve.
void LiveDebugValues::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  MachineFunctionPass::getAnalysisUsage(AU);
}

//===----------------------------------------------------------------------===//
//            Debug Range Extension Implementation
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
// Something to restore in the future.
//void LiveDebugValues::printVarLocInMBB(..)
#endif

SpillLoc
LiveDebugValues::extractSpillBaseRegAndOffset(const MachineInstr &MI) {
  assert(MI.hasOneMemOperand() &&
         "Spill instruction does not have exactly one memory operand?");
  auto MMOI = MI.memoperands_begin();
  const PseudoSourceValue *PVal = (*MMOI)->getPseudoValue();
  assert(PVal->kind() == PseudoSourceValue::FixedStack &&
         "Inconsistent memory operand in spill instruction");
  int FI = cast<FixedStackPseudoSourceValue>(PVal)->getFrameIndex();
  const MachineBasicBlock *MBB = MI.getParent();
  unsigned Reg;
  int Offset = TFI->getFrameIndexReference(*MBB->getParent(), FI, Reg);
  return {Reg, Offset};
}

/// End all previous ranges related to @MI and start a new range from @MI
/// if it is a DBG_VALUE instr.
bool LiveDebugValues::transferDebugValue(const MachineInstr &MI) {
  if (!MI.isDebugValue())
    return false;
  const DILocalVariable *Var = MI.getDebugVariable();
  const DIExpression *Expr = MI.getDebugExpression();
  const DILocation *DebugLoc = MI.getDebugLoc();
  const DILocation *InlinedAt = DebugLoc->getInlinedAt();
  assert(Var->isValidLocationForIntrinsic(DebugLoc) &&
         "Expected inlined-at fields to agree");

  DebugVariable V(Var, Expr, InlinedAt);

  auto *Scope = LS.findLexicalScope(MI.getDebugLoc().get());
  if (Scope == nullptr)
    return true; // handled it; by doing nothing

  const MachineOperand &MO = MI.getOperand(0);

  // MLocTracker needs to know that this register is read, even if it's only
  // read by a debug inst.
  if (MO.isReg() && MO.getReg() != 0)
    tracker->readReg(MO.getReg());

  if (vtracker) {
    if (MO.isReg()) {
      // Should read LocNo==0 on $noreg.
      ValueIDNum undef = {0, 0, LocIdx(0)};
      ValueIDNum ID = (MO.getReg()) ? tracker->readReg(MO.getReg()) : undef;
      vtracker->defVar(MI, ID);
    } else if (MI.getOperand(0).isImm() || MI.getOperand(0).isFPImm() ||
               MI.getOperand(0).isCImm()) {
      vtracker->defVar(MI, MI.getOperand(0));
    }
  }

  if (ttracker)
    ttracker->redefVar(MI);
  return true;
}

/// A definition of a register may mark the end of a range.
void LiveDebugValues::transferRegisterDef(
    MachineInstr &MI) {

  // Meta Instructions do not affect the debug liveness of any register they
  // define.
  if (MI.isMetaInstruction())
    return;

  MachineFunction *MF = MI.getMF();
  const TargetLowering *TLI = MF->getSubtarget().getTargetLowering();
  unsigned SP = TLI->getStackPointerRegisterToSaveRestore();

  // Find the regs killed by MI, and find regmasks of preserved regs.
  // Max out the number of statically allocated elements in `DeadRegs`, as this
  // prevents fallback to std::set::count() operations.
  SmallSet<uint32_t, 32> DeadRegs;
  SmallVector<const uint32_t *, 4> RegMasks;
  SmallVector<const MachineOperand *, 4> RegMaskPtrs;
  for (const MachineOperand &MO : MI.operands()) {
    // Determine whether the operand is a register def.
    if (MO.isReg() && MO.isDef() && MO.getReg() &&
        Register::isPhysicalRegister(MO.getReg()) &&
        !(MI.isCall() && MO.getReg() == SP)) {
      // Remove ranges of all aliased registers.
      for (MCRegAliasIterator RAI(MO.getReg(), TRI, true); RAI.isValid(); ++RAI)
        // FIXME: Can we break out of this loop early if no insertion occurs?
        DeadRegs.insert(*RAI);
    } else if (MO.isRegMask()) {
      RegMasks.push_back(MO.getRegMask());
      RegMaskPtrs.push_back(&MO);
    }
  }

  // Erase VarLocs which reside in one of the dead registers. For performance
  // reasons, it's critical to not iterate over the full set of open VarLocs.
  // Iterate over the set of dying/used regs instead.
  for (uint32_t DeadReg : DeadRegs)
    tracker->defReg(DeadReg, cur_bb, cur_inst);

  for (auto *MO : RegMaskPtrs) {
    tracker->writeRegMask(MO, cur_bb, cur_inst);
  }
}

bool LiveDebugValues::isSpillInstruction(const MachineInstr &MI,
                                         MachineFunction *MF) {
  // TODO: Handle multiple stores folded into one.
  if (!MI.hasOneMemOperand())
    return false;

  if (!MI.getSpillSize(TII) && !MI.getFoldedSpillSize(TII))
    return false; // This is not a spill instruction, since no valid size was
                  // returned from either function.

  return true;
}

bool LiveDebugValues::isLocationSpill(const MachineInstr &MI,
                                      MachineFunction *MF, unsigned &Reg) {
  if (!isSpillInstruction(MI, MF))
    return false;

  auto isKilledReg = [&](const MachineOperand MO, unsigned &Reg) {
    if (!MO.isReg() || !MO.isUse()) {
      Reg = 0;
      return false;
    }
    Reg = MO.getReg();
    return MO.isKill();
  };

  for (const MachineOperand &MO : MI.operands()) {
    // In a spill instruction generated by the InlineSpiller the spilled
    // register has its kill flag set.
    if (isKilledReg(MO, Reg))
      return true;
    if (Reg != 0) {
      // Check whether next instruction kills the spilled register.
      // FIXME: Current solution does not cover search for killed register in
      // bundles and instructions further down the chain.
      auto NextI = std::next(MI.getIterator());
      // Skip next instruction that points to basic block end iterator.
      if (MI.getParent()->end() == NextI)
        continue;
      unsigned RegNext;
      for (const MachineOperand &MONext : NextI->operands()) {
        // Return true if we came across the register from the
        // previous spill instruction that is killed in NextI.
        if (isKilledReg(MONext, RegNext) && RegNext == Reg)
          return true;
      }
    }
  }
  // Return false if we didn't find spilled register.
  return false;
}

Optional<SpillLoc>
LiveDebugValues::isRestoreInstruction(const MachineInstr &MI,
                                      MachineFunction *MF, unsigned &Reg) {
  if (!MI.hasOneMemOperand())
    return None;

  // FIXME: Handle folded restore instructions with more than one memory
  // operand.
  if (MI.getRestoreSize(TII)) {
    Reg = MI.getOperand(0).getReg();
    return extractSpillBaseRegAndOffset(MI);
  }
  return None;
}

/// A spilled register may indicate that we have to end the current range of
/// a variable and create a new one for the spill location.
/// A restored register may indicate the reverse situation.
/// Any change in location will be recorded in \p OpenRanges, and \p Transfers
/// if it is non-null.
bool LiveDebugValues::transferSpillOrRestoreInst(MachineInstr &MI) {
return false;
  MachineFunction *MF = MI.getMF();
  unsigned Reg;
  Optional<SpillLoc> Loc;

  LLVM_DEBUG(dbgs() << "Examining instruction: "; MI.dump(););

  // First, if there are any DBG_VALUEs pointing at a spill slot that is
  // written to, then close the variable location. The value in memory
  // will have changed.
  if (isSpillInstruction(MI, MF)) {
    Loc = extractSpillBaseRegAndOffset(MI);

    if (ttracker) {
      LocIdx mloc = tracker->getSpillMLoc(*Loc);
      if (mloc != 0)
        ttracker->clobberMloc(mloc, MI.getIterator());
    }
  }

  // Try to recognise spill and restore instructions that may create a new
  // variable location.
  if (isLocationSpill(MI, MF, Reg)) {
    Loc = extractSpillBaseRegAndOffset(MI);
    auto id = tracker->readReg(Reg);
    // If this is empty, produce an mphi.
    if (id.LocNo == 0)
      id = {cur_bb, 0, tracker->getRegMLoc(Reg)};
    tracker->setSpill(*Loc, id);
    assert(tracker->getSpillMLoc(*Loc) != 0);
    if (ttracker)
      ttracker->transferMlocs(tracker->getRegMLoc(Reg), tracker->getSpillMLoc(*Loc), MI.getIterator());
    for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
      tracker->defReg(*RAI, cur_bb, cur_inst);
  } else {
    if (!(Loc = isRestoreInstruction(MI, MF, Reg)))
      return false;
    auto id = tracker->readSpill(*Loc);
    if (id.LocNo != 0) {
      // XXX -- how do we go about tracking sub-values, one wonders?
      for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
        tracker->defReg(*RAI, cur_bb, cur_inst);
      // Override the reg we're restoring to. It's subregs go away. As they
      // do in old LDV.
      tracker->setReg(Reg, id);
      assert(tracker->getSpillMLoc(*Loc) != 0);
      if (ttracker)
        ttracker->transferMlocs(tracker->getSpillMLoc(*Loc), tracker->getRegMLoc(Reg), MI.getIterator());
    } else {
      // Well, def this register anyway.
      for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
        tracker->defReg(*RAI, cur_bb, cur_inst);
      // Make an mphi for the spill, to read it in the future.
      LocIdx l = tracker->getSpillMLoc(*Loc);
      id = {cur_bb, 0, l};
      tracker->setReg(Reg, id);
    }
  }
  return true;
}

/// If \p MI is a register copy instruction, that copies a previously tracked
/// value from one register to another register that is callee saved, we
/// create new DBG_VALUE instruction  described with copy destination register.
bool LiveDebugValues::transferRegisterCopy(MachineInstr &MI) {
  auto DestSrc = TII->isCopyInstr(MI);
  if (!DestSrc)
    return false;

  const MachineOperand *DestRegOp = DestSrc->Destination;
  const MachineOperand *SrcRegOp = DestSrc->Source;

  auto isCalleeSavedReg = [&](unsigned Reg) {
    for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
      if (CalleeSavedRegs.test(*RAI))
        return true;
    return false;
  };

  Register SrcReg = SrcRegOp->getReg();
  Register DestReg = DestRegOp->getReg();

  // We want to recognize instructions where destination register is callee
  // saved register. If register that could be clobbered by the call is
  // included, there would be a great chance that it is going to be clobbered
  // soon. It is more likely that previous register location, which is callee
  // saved, is going to stay unclobbered longer, even if it is killed.
  if (!isCalleeSavedReg(DestReg))
    return false;

  if (!SrcRegOp->isKill())
    return false;

  // We have to follow identity copies, as DbgEntityHistoryCalculator only
  // sees the defs.
  auto id = tracker->readReg(SrcReg);
  tracker->setReg(DestReg, id);
  if (ttracker)
    ttracker->transferMlocs(tracker->getRegMLoc(SrcReg), tracker->getRegMLoc(DestReg), MI.getIterator());

  if (SrcReg != DestReg)
    tracker->lolwipe(SrcReg);
  return true;
}

/// Accumulate a mapping between each DILocalVariable fragment and other
/// fragments of that DILocalVariable which overlap. This reduces work during
/// the data-flow stage from "Find any overlapping fragments" to "Check if the
/// known-to-overlap fragments are present".
/// \param MI A previously unprocessed DEBUG_VALUE instruction to analyze for
///           fragment usage.
void LiveDebugValues::accumulateFragmentMap(MachineInstr &MI) {
  DebugVariable MIVar(MI.getDebugVariable(), MI.getDebugExpression(),
                      MI.getDebugLoc()->getInlinedAt());
  FragmentInfo ThisFragment = MIVar.getFragmentOrDefault();

  // If this is the first sighting of this variable, then we are guaranteed
  // there are currently no overlapping fragments either. Initialize the set
  // of seen fragments, record no overlaps for the current one, and return.
  auto SeenIt = SeenFragments.find(MIVar.getVariable());
  if (SeenIt == SeenFragments.end()) {
    SmallSet<FragmentInfo, 4> OneFragment;
    OneFragment.insert(ThisFragment);
    SeenFragments.insert({MIVar.getVariable(), OneFragment});

    OverlapFragments.insert({{MIVar.getVariable(), ThisFragment}, {}});
    return;
  }

  // If this particular Variable/Fragment pair already exists in the overlap
  // map, it has already been accounted for.
  auto IsInOLapMap =
      OverlapFragments.insert({{MIVar.getVariable(), ThisFragment}, {}});
  if (!IsInOLapMap.second)
    return;

  auto &ThisFragmentsOverlaps = IsInOLapMap.first->second;
  auto &AllSeenFragments = SeenIt->second;

  // Otherwise, examine all other seen fragments for this variable, with "this"
  // fragment being a previously unseen fragment. Record any pair of
  // overlapping fragments.
  for (auto &ASeenFragment : AllSeenFragments) {
    // Does this previously seen fragment overlap?
    if (DIExpression::fragmentsOverlap(ThisFragment, ASeenFragment)) {
      // Yes: Mark the current fragment as being overlapped.
      ThisFragmentsOverlaps.push_back(ASeenFragment);
      // Mark the previously seen fragment as being overlapped by the current
      // one.
      auto ASeenFragmentsOverlaps =
          OverlapFragments.find({MIVar.getVariable(), ASeenFragment});
      assert(ASeenFragmentsOverlaps != OverlapFragments.end() &&
             "Previously seen var fragment has no vector of overlaps");
      ASeenFragmentsOverlaps->second.push_back(ThisFragment);
    }
  }

  AllSeenFragments.insert(ThisFragment);
}

/// This routine creates OpenRanges.
void LiveDebugValues::process(MachineInstr &MI) {
  if (transferDebugValue(MI))
    return;
  if (transferRegisterCopy(MI))
    return;
  if (transferSpillOrRestoreInst(MI))
    return;
  transferRegisterDef(MI);
}

void LiveDebugValues::produce_mloc_transfer_function(MachineFunction &MF,
                           std::vector<mloc_transfert> &MLocTransfer,
                           unsigned MaxNumBlocks)
{
  std::vector<BitVector> BlockMasks;
  BlockMasks.resize(MaxNumBlocks);

  unsigned BVWords = MachineOperand::getRegMaskSize(TRI->getNumRegs());
  for (auto &BV : BlockMasks)
    BV.resize(TRI->getNumRegs(), true);

  // Step through all instructions and inhale the transfer function.
  for (auto &MBB : MF) {
    cur_bb = MBB.getNumber();
    cur_inst = 1;

    tracker->reset();
    tracker->setMPhis(cur_bb);
    for (auto &MI : MBB) {
      process(MI);
      // Also accumulate fragment map.
      if (MI.isDebugValue())
        accumulateFragmentMap(MI);
      ++cur_inst;
    }

    // Look at tracker: still has input phi means no assignment. Produce
    // a mapping if there's a movement.
    for (unsigned IdxNum = 1; IdxNum < tracker->getNumLocs(); ++IdxNum) {
      LocIdx Idx = LocIdx(IdxNum);
      VarLocPos P = tracker->getVarLocPos(Idx);
      if (P.ID.InstNo == 0 && P.ID.LocNo == P.CurrentLoc)
        continue;

      MLocTransfer[cur_bb][Idx] = P.ID;
    }

    // Accumulate any bitmask operands.
    for (auto &P : tracker->Masks) {
      BlockMasks[cur_bb].clearBitsNotInMask(P.first->getRegMask(), BVWords);
    }
  }

  const TargetLowering *TLI = MF.getSubtarget().getTargetLowering();
  unsigned SP = TLI->getStackPointerRegisterToSaveRestore();
  BitVector UsedRegs(TRI->getNumRegs());
  for (auto &P : tracker->LocIdxToLocID) {
    if (P.first == 0 || P.second >= TRI->getNumRegs() || P.second == SP)
      continue;
    UsedRegs.set(P.second);
  }

  // For each block that we looked at, are there any clobbered registers that
  // are used, and that don't appear as 'clobbered' in the transfer func?
  // Overwrite them. XXX, this doesn't account for setting a reg and then
  // clobbering it afterwards, although I guess then the reg would be known
  // about?
  for (unsigned int I = 0; I < MaxNumBlocks; ++I) {
    BitVector &BV = BlockMasks[I];
    BV.flip();
    BV &= UsedRegs;
    // This produces all the bits that we clobber, but also use. Check that
    // they're all clobbered or at least set in the designated transfer
    // elem.
    for (unsigned Bit : BV.set_bits()) {
      unsigned ID = tracker->getLocID(Bit, false);
      LocIdx Idx = tracker->LocIDToLocIdx[ID];
      assert(Idx != 0);
      ValueIDNum &ValueID = MLocTransfer[I][Idx];
      if (ValueID.BlockNo == I && ValueID.InstNo == 0)
        // it was left as live-through. Set it to clobbered.
        ValueID = ValueIDNum{0, 0, LocIdx(0)};
    }
  }
}

/// This routine joins the analysis results of all incoming edges in @MBB by
/// inserting a new DBG_VALUE instruction at the start of the @MBB - if the same
/// source variable in all the predecessors of @MBB reside in the same location.
bool LiveDebugValues::mloc_join(
    MachineBasicBlock &MBB,
    SmallPtrSet<const MachineBasicBlock *, 16> &Visited,
    uint64_t **OutLocs, uint64_t *InLocs) {
  LLVM_DEBUG(dbgs() << "join MBB: " << MBB.getNumber() << "\n");
  bool Changed = false;

  // Collect predecessors that have been visited.
  SmallVector<const MachineBasicBlock *, 8> BlockOrders;
  for (auto p : MBB.predecessors()) {
    if (Visited.count(p)) {
      BlockOrders.push_back(p);
    }
  }

  auto Cmp = [&](const MachineBasicBlock *A, const MachineBasicBlock *B) {
   return BBToOrder.find(A)->second < BBToOrder.find(B)->second;
  };
  llvm::sort(BlockOrders.begin(), BlockOrders.end(), Cmp);

  // Skip entry block.
  if (BlockOrders.size() == 0)
    return false;

    // Step through all predecessors and detect disagreements.
  unsigned this_rpot = BBToOrder.find(&MBB)->second;
  for (unsigned Idx = 1; Idx < tracker->getNumLocs(); ++Idx) {
    uint64_t base = OutLocs[BlockOrders[0]->getNumber()][Idx];
    bool disagree = false;
    bool pred_disagree = false;
    for (auto *MBB : BlockOrders) { // xxx loops around itself.
      if (base != OutLocs[MBB->getNumber()][Idx]) {
        disagree = true;
        if (BBToOrder.find(MBB)->second < this_rpot) // might be self b/e
          pred_disagree = true;
      }
    }

    bool over_ride = false;
    if (disagree && !pred_disagree && ValueIDNum::fromU64(InLocs[Idx]).LocNo != 0) {
      // It's only the backedges that disagree. Consider demoting. Order is
      // that non-phis have the minimum priority, and phis "closer" to this
      // one.
      ValueIDNum base_id = ValueIDNum::fromU64(base);
      ValueIDNum inloc_id = ValueIDNum::fromU64(InLocs[Idx]);
      unsigned base_block = base_id.BlockNo + 1;
      if (base_id.InstNo != 0)
        base_block = 0;
      unsigned inloc_block = inloc_id.BlockNo + 1;
      if (inloc_id.InstNo != 0)
        inloc_block = 0;
      if (base_block > inloc_block) {
        // Override.
        over_ride = true;
      }
    }

    // Generate a phi...
    ValueIDNum PHI = {(uint64_t)MBB.getNumber(), 0, LocIdx(Idx)};
    uint64_t NewVal = (disagree && !over_ride) ? PHI.asU64() : base;
    if (InLocs[Idx] != NewVal) {
      Changed |= true;
      InLocs[Idx] = NewVal;
    }
  }

  // Uhhhhhh, reimplement NumInserted and NumRemoved pls.
  return Changed;
}

void LiveDebugValues::mloc_dataflow(uint64_t **MInLocs,
    uint64_t **MOutLocs, std::vector<mloc_transfert> &MLocTransfer) {
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Worklist, Pending;

  for (unsigned int I = 0; I < BBToOrder.size(); ++I)
    Worklist.push(I);

  tracker->reset();
  // Set inlocs for entry block,
  tracker->setMPhis(0);
  for (unsigned Idx = 1; Idx < tracker->getNumLocs(); ++Idx) {
    auto VLP = tracker->getVarLocPos(LocIdx(Idx));
    uint64_t ID = VLP.ID.asU64();
    MInLocs[0][Idx] = ID;
  }

  SmallPtrSet<const MachineBasicBlock *, 16> Visited;
  while (!Worklist.empty() || !Pending.empty()) {
    // We track what is on the pending worklist to avoid inserting the same
    // thing twice.  We could avoid this with a custom priority queue, but this
    // is probably not worth it.
    SmallPtrSet<MachineBasicBlock *, 16> OnPending;
    SmallVector<std::pair<LocIdx, ValueIDNum>, 32> toremap;
    while (!Worklist.empty()) {
      MachineBasicBlock *MBB = OrderToBB[Worklist.top()];
      cur_bb = MBB->getNumber();
      Worklist.pop();

      bool InLocsChanged = mloc_join(*MBB, Visited, MOutLocs, MInLocs[cur_bb]);
      InLocsChanged |= Visited.insert(MBB).second;

      // Don't examine transfer function if we've visited this loc at least
      // once, and inlocs haven't changed.
      if (!InLocsChanged)
        continue;

      // Rather than touch all insts again, read and then reset locations
      // in the transfer function.
      tracker->loadFromArray(MInLocs[cur_bb], cur_bb);
      toremap.clear();
      for (auto &P : MLocTransfer[cur_bb]) {
        ValueIDNum NewID = {0, 0, LocIdx(0)};
        if (P.second.BlockNo == cur_bb && P.second.InstNo == 0) {
          // This is a movement of whatever was live in. Read it.
          VarLocPos Pos = tracker->getVarLocPos(P.second.LocNo);
          NewID = Pos.ID;
        } else {
          // It's a def. (Has to be a def in this BB, or nullloc).
          // Just set it.
          assert(P.second.BlockNo == cur_bb || P.second.LocNo == 0);
          NewID = P.second;
        }
        toremap.push_back(std::make_pair(P.first, NewID));
      }

      // Commit the transfer function changes into mloc tracker.
      for (auto &P : toremap)
        tracker->setMLoc(P.first, P.second);

      // Now copy out-locs from mloc tracker into out-loc vector, checking
      // whether changes have occurred. These changes can have come from both
      // the transfer function, and mloc_join.
      bool OLChanged = false;
      for (unsigned Idx = 1; Idx < tracker->getNumLocs(); ++Idx) {
        auto VLP = tracker->getVarLocPos(LocIdx(Idx));
        uint64_t ID = VLP.ID.asU64();
        OLChanged |= MOutLocs[cur_bb][Idx] != ID;
        MOutLocs[cur_bb][Idx] = ID;
      }

      tracker->reset();

      // No need to examine successors again if out-locs didn't change.
      if (!OLChanged)
        continue;

      for (auto s : MBB->successors())
        if (OnPending.insert(s).second)
          Pending.push(BBToOrder[s]);
    }
    Worklist.swap(Pending);
    // At this point, pending must be empty, since it was just the empty
    // worklist
    assert(Pending.empty() && "Pending should be empty");
  }
}

bool LiveDebugValues::vloc_join_location(MachineBasicBlock &MBB,
                        ValueRec &InLoc,
                        ValueRec &OLoc, uint64_t *InLocOutLocs,
                        uint64_t *OLOutLocs,
                        const LiveIdxT::mapped_type PrevInLocs, // ptr
                        const DebugVariable &CurVar, bool ThisIsABackEdge)
{
  unsigned cur_bb = MBB.getNumber();
  bool EarlyBail = false;

  auto FindLocInLocs = [&](uint64_t *OutLocs, const ValueIDNum &ID) -> LocIdx {
    unsigned NumLocs = tracker->getNumLocs();
    LocIdx theloc = LocIdx(0);
    for (unsigned i = 0; i < NumLocs; ++i) {
      if (OutLocs[i] == ID.asU64()) {
        if (theloc != 0) {
          // Prefer non-spills
          if (tracker->isSpill(theloc))
            theloc = LocIdx(i);
        } else {
          theloc = LocIdx(i);
        }
      }
    }
    // It's possible that that value simply isn't availble, coming out of the
    // designated block.
    return theloc;
  };
  auto FindInInLocs = [&](const ValueIDNum &ID) -> LocIdx {
    return FindLocInLocs(InLocOutLocs, ID);
  };
  auto FindInOLocs = [&](const ValueIDNum &ID) -> LocIdx {
    return FindLocInLocs(OLOutLocs, ID);
  };

  // Different kinds? Definite no.
  EarlyBail |= InLoc.Kind != OLoc.Kind;

  // Trying to join constants is very simple. Plain join on the constant
  // value.
  EarlyBail |=
     (InLoc.Kind == OLoc.Kind && InLoc.Kind == ValueRec::Const &&
      !InLoc.MO->isIdenticalTo(*OLoc.MO));

  // Meta disagreement -> bail early.
  EarlyBail |= (InLoc.meta != OLoc.meta);

  // Clobbered location -> bail early.
  EarlyBail |=
     (InLoc.Kind == OLoc.Kind && InLoc.Kind == ValueRec::Def &&
      OLoc.ID.LocNo == 0);

  // Bail out if early bail signalled.
  if (EarlyBail) {
    return false;
  } else if (InLoc.Kind == ValueRec::Const) {
    // If both are constants, we've satisfied constraints.
    return true;
  }

  // This is a join for "values". Two important facts: is this a
  // backedge, and does the machine-location we're joining on contain
  // an mphi? (InlocsIt will be the mphi if there is one).
  assert(InLoc.Kind == ValueRec::Def);
  ValueIDNum &InLocsID = InLoc.ID;
  ValueIDNum &OLID = OLoc.ID;
  bool ThisIsAnMPHI = InLocsID.BlockNo == cur_bb && InLocsID.InstNo == 0;

  // Pick out whether the OLID is in the backedge location or not.
  LocIdx OLIdx = FindInOLocs(OLID);
  if (OLIdx == 0 && OLID.BlockNo == cur_bb && OLID.InstNo == 0)
    OLIdx = OLID.LocNo; // We've previously made this an mphi.

  // If it isn't, this location is invalidated _in_ the block on the
  // other end of the backedge.
  if (OLOutLocs[OLIdx] != OLID.asU64())
    return false;

  // Everything is massively different for backedges. Try not-be's first.
  if (!ThisIsABackEdge) {
    // Identical? Then we simply agree. Unless there's an mphi, in which
    // case we risk the mloc values not lining up being missed. Apply
    // harder checks to force this to become an mphi location, or croak.
    if (InLoc == OLoc && !ThisIsAnMPHI)
      return true;

    // If we're non-identical and there's no mphi, definitely can't merge.
    if (InLoc != OLoc && !ThisIsAnMPHI)
      return false;

    // Otherwise, we're definitely an mphi, and need to prove that the
    // location from olit goes into it. Because we're an mphi, we know
    // our location...
    LocIdx InLocIdx = InLocsID.LocNo;
    // Also necessary: the vloc out-loc for the edge matches the mloc
    // out-loc.
    bool HasMOutLoc = OLOutLocs[InLocIdx] == OLID.asU64();
    if (!HasMOutLoc)
      // They conflict and are in the wrong location. Incompatible.
      return false;
    return true;
  }

  LocIdx Idx = FindInInLocs(InLocsID);
  if (Idx == 0 && InLocsID.BlockNo == cur_bb && InLocsID.InstNo == 0)
    Idx = InLocsID.LocNo; // We've previously made this an mphi.

  // OK, the value is fed back around. If it's the same, it must be
  // the same in the same location.
  if (InLocsID == OLID) {
    if (Idx != OLIdx)
      return false;
    return true;
  }

  // Values aren't equal: filter for they're coming back around to an
  // mphi starting at this block.
  if (Idx == OLIdx && ThisIsAnMPHI)
    return true;

  // We're not identical, values are merging and there's no an mphi
  // starting at this block. Check for something where we're being
  // overridden by an mphi found earlier in the tree.

  // consider overriding.
  auto ILS_It = PrevInLocs->find(CurVar);
  if (ILS_It == PrevInLocs->end() || ILS_It->second.Kind != ValueRec::Def)
    // First time around, if there are disagreements, they won't
    // be due to back edges, thus it's immediately fatal.
    return false;

  ValueIDNum &ILS_ID = ILS_It->second.ID;
  unsigned NewInOrder = (InLocsID.InstNo) ? 0 : InLocsID.BlockNo + 1;
  unsigned OldOrder = (ILS_ID.InstNo) ? 0 : ILS_ID.BlockNo + 1;
  if (OldOrder >= NewInOrder)
    return false;

  return true;
}

bool LiveDebugValues::vloc_join(
  MachineBasicBlock &MBB, LiveIdxT &VLOCOutLocs,
   LiveIdxT &VLOCInLocs,
   SmallPtrSet<const MachineBasicBlock *, 16> *VLOCVisited,
   unsigned cur_bb,
   const SmallSet<DebugVariable, 4> &AllVars,
   uint64_t **MInLocs, uint64_t **MOutLocs,
  SmallPtrSet<const MachineBasicBlock *, 8> &NonAssignBlocks) {
   
  if (NonAssignBlocks.count(&MBB) == 0) {
    // Wipe all inlocs. By never assigning to them.
    if (VLOCVisited)
      return true;
    return false;
  }

  LLVM_DEBUG(dbgs() << "join MBB: " << MBB.getNumber() << "\n");
  bool Changed = false;

  DenseMap<DebugVariable, ValueRec> InLocsT;
  SmallSet<DebugVariable, 8> Disagreements;

  auto ILSIt = VLOCInLocs.find(&MBB);
  assert(ILSIt != VLOCInLocs.end());
  auto &ILS = *ILSIt->second;

  auto FindLocOfDef = [&](unsigned BBNum, const ValueIDNum &ID) -> LocIdx {
    unsigned NumLocs = tracker->getNumLocs();
    uint64_t *OutLocs = MOutLocs[BBNum];
    LocIdx theloc = LocIdx(0);
    for (unsigned i = 0; i < NumLocs; ++i) {
      if (OutLocs[i] == ID.asU64()) {
        if (theloc != 0) {
          // Prefer non-spills
          if (tracker->isSpill(theloc))
            theloc = LocIdx(i);
        } else {
          theloc = LocIdx(i);
        }
      }
    }
    // It's possible that that value simply isn't availble, coming out of the
    // designated block.
    return theloc;
  };

  // Order predecessors by RPOT order. Fundemental right now.
  SmallVector<MachineBasicBlock *, 8> BlockOrders;
  for (auto p : MBB.predecessors())
    BlockOrders.push_back(p);

  auto Cmp = [&](MachineBasicBlock *A, MachineBasicBlock *B) {
    return BBToOrder[A] < BBToOrder[B];
  };

  llvm::sort(BlockOrders.begin(), BlockOrders.end(), Cmp);
  unsigned this_rpot = BBToOrder[&MBB];

  // For all predecessors of this MBB, find the set of VarLocs that
  // can be joined.
  int NumVisited = 0;
  unsigned FirstVisited = 0;
  for (auto p : BlockOrders) {
    // Ignore backedges if we have not visited the predecessor yet. As the
    // predecessor hasn't yet had locations propagated into it, most locations
    // will not yet be valid, so treat them as all being uninitialized and
    // potentially valid. If a location guessed to be correct here is
    // invalidated later, we will remove it when we revisit this block.
    if (VLOCVisited && !VLOCVisited->count(p)) {
      LLVM_DEBUG(dbgs() << "  ignoring unvisited pred MBB: " << p->getNumber()
                        << "\n");
      continue;
    }
    auto OL = VLOCOutLocs.find(p);
    // Join is null in case of empty OutLocs from any of the pred.
    if (OL == VLOCOutLocs.end()) {
      InLocsT.clear();
      break;
    }

    // Just copy over the Out locs to incoming locs for the first visited
    // predecessor, and for all other predecessors join the Out locs.
    if (!NumVisited) {
      InLocsT = *OL->second;
      FirstVisited = p->getNumber();

// XXX maaayyybbeeee downgrade to an mphi
for (auto &It : InLocsT) {
  // Where does it come out...
  if (It.second.Kind != ValueRec::Def)
    continue;
  LocIdx Idx = FindLocOfDef(FirstVisited, It.second.ID);
  if (Idx == 0)
    continue;
  // Is that what comes in?
  ValueIDNum LiveInID = ValueIDNum::fromU64(MInLocs[cur_bb][Idx]);
  if (It.second.ID != LiveInID) {
    // Ooops. It became an mphi. Convert it to one and check other things later.
    assert(LiveInID.BlockNo == cur_bb && LiveInID.InstNo == 0);
    It.second.ID = LiveInID;
  }
}

    } else {
      // XXX insert join here.
      for (auto &Var : AllVars) {
        auto InLocsIt = InLocsT.find(Var);
        auto OLIt = OL->second->find(Var);

        // Regardless of what's being joined in, an empty predecessor means
        // there can be no incoming location here.
        if (InLocsIt == InLocsT.end())
          continue;

        if (OLIt == OL->second->end()) {
          InLocsT.erase(InLocsIt);
          continue;
        }


        bool ThisIsABackEdge = this_rpot <= BBToOrder[p];
        bool joins = vloc_join_location(MBB, InLocsIt->second, 
                        OLIt->second, MOutLocs[FirstVisited],
                        MOutLocs[p->getNumber()], &ILS, InLocsIt->first,
                        ThisIsABackEdge);

        if (!joins)
          InLocsT.erase(InLocsIt);
      }
    }

    // xXX jmorse deleted debug statement

    NumVisited++;
  }

  // As we are processing blocks in reverse post-order we
  // should have processed at least one predecessor, unless it
  // is the entry block which has no predecessor.
#if 0
  assert((NumVisited || MBB.pred_empty()) &&
         "Should have processed at least one predecessor");
#endif

  Changed = ILS != InLocsT;
  if (Changed)
    ILS = std::move(InLocsT);
  // Uhhhhhh, reimplement NumInserted and NumRemoved pls.
  return Changed;
}

void LiveDebugValues::vloc_dataflow(
    const LexicalScope *Scope,
    const SmallSet<DebugVariable, 4> &VarsWeCareAbout,
    SmallPtrSetImpl<MachineBasicBlock *> &AssignBlocks,
    SmallVectorImpl<SmallVector<std::pair<DebugVariable, ValueRec>, 8>> &Output,
    uint64_t **MOutLocs, uint64_t **MInLocs,
    SmallVectorImpl<VLocTracker> &AllTheVLocs) {
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Worklist, Pending;

  SmallPtrSet<const MachineBasicBlock *, 8> LBlocks;
  SmallVector<MachineBasicBlock *, 8> BlockOrders;
  auto Cmp = [&](MachineBasicBlock *A, MachineBasicBlock *B) {
    return BBToOrder[A] < BBToOrder[B];
  };

  // Determine which blocks we're dealing with.
  assert(VarsWeCareAbout.size() != 0);
  auto AVar = *VarsWeCareAbout.begin();
  DebugLoc DL =
      DebugLoc::get(0, 0, AVar.getVariable()->getScope(), AVar.getInlinedAt());

  LS.getMachineBasicBlocks(DL.get(), LBlocks);
  SmallPtrSet<const MachineBasicBlock *, 8> NonAssignBlocks = LBlocks;

  // Also any blocks that contain a DBG_VALUE.
  LBlocks.insert(AssignBlocks.begin(), AssignBlocks.end());

  // Add all artifical blocks. This might be inefficient; lets deal with
  // that later. They won't contribute a lot unless they connect to a
  // meaningful non-artificial block.
  LBlocks.insert(ArtificialBlocks.begin(), ArtificialBlocks.end());
  NonAssignBlocks.insert(ArtificialBlocks.begin(), ArtificialBlocks.end());

  // Single block scope: not interesting! No propagation at all. Note that
  // this could probably go above ArtificialBlocks without damage, but
  // that then produces output differences from original-live-debug-values,
  // which propagates from a single block into many artificial ones.
  if (LBlocks.size() == 1)
    return;

  // Picks out their RPOT order and sort it.
  for (auto *MBB : LBlocks)
    BlockOrders.push_back(const_cast<MachineBasicBlock *>(MBB));

  llvm::sort(BlockOrders.begin(), BlockOrders.end(), Cmp);
  unsigned NumBlocks = BlockOrders.size();

  std::vector<DenseMap<DebugVariable, ValueRec>> LiveIns, LiveOuts;
  LiveIns.resize(NumBlocks);
  LiveOuts.resize(NumBlocks);
  LiveIdxT LiveOutIdx, LiveInIdx;
  for (unsigned I = 0; I < NumBlocks; ++I) {
    LiveOutIdx[BlockOrders[I]] = &LiveOuts[I];
    LiveInIdx[BlockOrders[I]] = &LiveIns[I];
  }

  for (auto *MBB : BlockOrders)
    Worklist.push(BBToOrder[MBB]);

  bool firsttrip = true;
  SmallPtrSet<const MachineBasicBlock *, 16> VLOCVisited;
  while (!Worklist.empty() || !Pending.empty()) {
    SmallPtrSet<MachineBasicBlock *, 16> OnPending;
    while (!Worklist.empty()) {
      auto *MBB = OrderToBB[Worklist.top()];
      cur_bb = MBB->getNumber(); // XXX ldv state
      Worklist.pop();

      // Join locations from predecessors.
      bool InlocsChanged = vloc_join(*MBB, LiveOutIdx, LiveInIdx,
                                     (firsttrip) ? &VLOCVisited : nullptr,
                                     cur_bb, VarsWeCareAbout, MInLocs, MOutLocs,
                                     NonAssignBlocks);

      // Always explore transfer function if inlocs changed, or if we've not
      // visited this block before.
      InlocsChanged |= VLOCVisited.insert(MBB).second;
      if (!InlocsChanged)
        continue;

      // Do transfer function.
      // DenseMap copy.
      DenseMap<DebugVariable, ValueRec> Cpy = *LiveInIdx[MBB];
      auto &vtracker = AllTheVLocs[MBB->getNumber()];
      for (auto &Transfer : vtracker.Vars) {
        // Is this var we're mangling in this scope?
        if (VarsWeCareAbout.count(Transfer.first)) {
          // Erase on empty transfer (DBG_VALUE $noreg).
          if (Transfer.second.Kind == ValueRec::Def &&
              Transfer.second.ID.LocNo == 0)
            Cpy.erase(Transfer.first);
          else
            Cpy[Transfer.first] = Transfer.second;
        }
      }

      // Commit newly calculated live-outs, nothing whether they changed.
      bool OLChanged = Cpy != *LiveOutIdx[MBB];
      *LiveOutIdx[MBB] = Cpy;

      // If they haven't changed, there's no need to explore further.
      if (!OLChanged)
        continue;

      // Ignore out of scope successors and those already on the list. All
      // others should be on the pending list next time around.
      for (auto s : MBB->successors())
        if (LiveInIdx.find(s) != LiveInIdx.end() && OnPending.insert(s).second)
          Pending.push(BBToOrder[s]);
    }
    Worklist.swap(Pending);
    assert(Pending.empty());
    firsttrip = false;
  }

  // Dataflow done. Now what? Save live-ins.
  for (unsigned I = 0; I < LiveIns.size(); ++I) {
    auto &VarMap = LiveIns[I];
    auto *MBB = BlockOrders[I];
    for (auto &P : VarMap) {
      Output[MBB->getNumber()].push_back(P);
    }
  }

  BlockOrders.clear();
  LBlocks.clear();
}

void LiveDebugValues::dump_mloc_transfer(const mloc_transfert &mloc_transfer) const {
  for (auto &P : mloc_transfer) {
    std::string foo = tracker->LocIdxToName(P.first);
    std::string bar = tracker->IDAsString(P.second);
    dbgs() << "Loc " << foo << " --> " << bar << "\n";
  }
}

/// Calculate the liveness information for the given machine function and
/// extend ranges across basic blocks.
bool LiveDebugValues::ExtendRanges(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "\nDebug Range Extension\n");

  std::vector<mloc_transfert> MLocTransfer;
  SmallVector<VLocTracker, 8> vlocs;
  SmallVector<SmallVector<std::pair<DebugVariable, ValueRec>, 8>, 16> SavedLiveIns;

  int MaxNumBlocks = -1;
  for (auto &MBB : MF)
    MaxNumBlocks = std::max(MBB.getNumber(), MaxNumBlocks);
  assert(MaxNumBlocks >= 0);
  ++MaxNumBlocks;

  MLocTransfer.resize(MaxNumBlocks);
  vlocs.resize(MaxNumBlocks);
  SavedLiveIns.resize(MaxNumBlocks);

  auto hasNonArtificialLocation = [](const MachineInstr &MI) -> bool {
    if (const DebugLoc &DL = MI.getDebugLoc())
      return DL.getLine() != 0;
    return false;
  };
  for (auto &MBB : MF)
    if (none_of(MBB.instrs(), hasNonArtificialLocation))
      ArtificialBlocks.insert(&MBB);

  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  unsigned int RPONumber = 0;
  for (auto RI = RPOT.begin(), RE = RPOT.end(); RI != RE; ++RI) {
    OrderToBB[RPONumber] = *RI;
    BBToOrder[*RI] = RPONumber;
    ++RPONumber;
  }

  produce_mloc_transfer_function(MF, MLocTransfer, MaxNumBlocks);

  // Huurrrr. Store liveouts in a massive array.
  uint64_t **MOutLocs = new uint64_t *[MaxNumBlocks];
  uint64_t **MInLocs = new uint64_t *[MaxNumBlocks];
  unsigned NumLocs = tracker->getNumLocs();
  for (int i = 0; i < MaxNumBlocks; ++i) {
    MOutLocs[i] = new uint64_t[NumLocs];
    memset(MOutLocs[i], 0xFF, sizeof(uint64_t) * NumLocs);
    MInLocs[i] = new uint64_t[NumLocs];
    memset(MInLocs[i], 0, sizeof(uint64_t) * NumLocs);
  }

  mloc_dataflow(MInLocs, MOutLocs, MLocTransfer);

  // Accumulate things into the vloc tracker.
  for (auto RI = RPOT.begin(), RE = RPOT.end(); RI != RE; ++RI) {
    cur_bb = (*RI)->getNumber();
    auto *MBB = *RI;
    vtracker = &vlocs[cur_bb];
    vtracker->MBB = MBB;
    tracker->loadFromArray(MInLocs[cur_bb], cur_bb);
    cur_inst = 1;
    for (auto &MI : *MBB) {
      process(MI);
      ++cur_inst;
    }
    tracker->reset();
  }

  // Produce a set of all variables.
  DenseMap<DebugVariable, unsigned> AllVarsNumbering;
  DenseMap<const LexicalScope *, SmallSet<DebugVariable, 4>> ScopeToVars;
  DenseMap<const LexicalScope *, SmallPtrSet<MachineBasicBlock *, 4>> ScopeToBlocks;
  // To match old LDV, enumerate variables in RPOT order.
  for (auto RI = RPOT.begin(), RE = RPOT.end(); RI != RE; ++RI) {
    auto *vtracker = &vlocs[(*RI)->getNumber()];
    for (auto &idx : vtracker->Vars) {
      const auto &Var = idx.first;
      DebugLoc DL = DebugLoc::get(0, 0, Var.getVariable()->getScope(), Var.getInlinedAt());
      auto *Scope = LS.findLexicalScope(DL.get());

      // No insts in scope -> shouldn't have been recorded.
      assert(Scope != nullptr);

      AllVarsNumbering.insert(std::make_pair(Var, AllVarsNumbering.size()));
      ScopeToVars[Scope].insert(Var);
      ScopeToBlocks[Scope].insert(vtracker->MBB);
    }
  }

  // OK. Iterate over scopes: there might be something to be said for
  // ordering them by size/locality, but that's for the future.
  for (auto &P : ScopeToVars) {
    vloc_dataflow(P.first, P.second, ScopeToBlocks[P.first],
                  SavedLiveIns, MOutLocs, MInLocs, vlocs);
  }

  // mloc argument only needs the posish -> spills map and the like.
  ttracker = new TransferTracker(TII, tracker, MF);

  for (MachineBasicBlock &MBB : MF) {
    unsigned bbnum = MBB.getNumber();
    tracker->reset();
    tracker->loadFromArray(MInLocs[bbnum], bbnum);
    ttracker->loadInlocs(MBB, MInLocs[bbnum], SavedLiveIns[MBB.getNumber()], bbnum, NumLocs);

    cur_bb = bbnum;
    cur_inst = 1;
    for (auto &MI : MBB) {
      process(MI);
      ++cur_inst;
    }
  }

  // XXX remove earlier LiveIn ordering and see whether it's needed now.
  auto OrderDbgValues = [&](const MachineInstr *A, const MachineInstr *B) -> bool{
    DebugVariable VarA(A->getDebugVariable(), A->getDebugExpression(),
                      A->getDebugLoc()->getInlinedAt());
    DebugVariable VarB(B->getDebugVariable(), B->getDebugExpression(),
                      B->getDebugLoc()->getInlinedAt());
    return AllVarsNumbering.find(VarA)->second < AllVarsNumbering.find(VarB)->second;
  };

  for (auto &P : ttracker->Transfers) {
    llvm::sort(P.insts.begin(), P.insts.end(), OrderDbgValues);
    if (P.MBB) {
      MachineBasicBlock &MBB = *P.MBB;
      for (auto *MI : P.insts) {
        MBB.insert(P.pos, MI);
      }
    } else {
      MachineBasicBlock &MBB = *P.pos->getParent();
      for (auto *MI : P.insts) {
        MBB.insertAfter(P.pos, MI);
      }
    }
  }

  for (int Idx = 0; Idx < MaxNumBlocks; ++Idx) {
    delete[] MOutLocs[Idx];
    delete[] MInLocs[Idx];
  }
  delete[] MOutLocs;
  delete[] MInLocs;

  return ttracker->Transfers.size() != 0;
}

bool LiveDebugValues::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getFunction().getSubprogram())
    // LiveDebugValues will already have removed all DBG_VALUEs.
    return false;

  // Skip functions from NoDebug compilation units.
  if (MF.getFunction().getSubprogram()->getUnit()->getEmissionKind() ==
      DICompileUnit::NoDebug)
    return false;

  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();
  TFI = MF.getSubtarget().getFrameLowering();
  TFI->getCalleeSaves(MF, CalleeSavedRegs);
  LS.initialize(MF);

  tracker = new MLocTracker(MF, *TII, *TRI, *MF.getSubtarget().getTargetLowering());
  vtracker = nullptr;
  ttracker = nullptr;

  bool Changed = ExtendRanges(MF);
  delete tracker;
  vtracker = nullptr;
  ttracker = nullptr;

  ArtificialBlocks.clear();
  OrderToBB.clear();
  BBToOrder.clear();

  return Changed;
}
