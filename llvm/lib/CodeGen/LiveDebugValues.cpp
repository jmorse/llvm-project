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

// If @MI is a DBG_VALUE with debug value described by a defined
// register, returns the number of this register. In the other case, returns 0.
static Register isDbgValueDescribedByReg(const MachineInstr &MI) {
  assert(MI.isDebugValue() && "expected a DBG_VALUE");
  assert(MI.getNumOperands() == 4 && "malformed DBG_VALUE");
  // If location of variable is described using a register (directly
  // or indirectly), this register is always a first operand.
  return MI.getOperand(0).isReg() ? MI.getOperand(0).getReg() : Register();
}

/// If \p Op is a stack or frame register return true, otherwise return false.
/// This is used to avoid basing the debug entry values on the registers, since
/// we do not support it at the moment.
static bool isRegOtherThanSPAndFP(const MachineOperand &Op,
                                  const MachineInstr &MI,
                                  const TargetRegisterInfo *TRI) {
  if (!Op.isReg())
    return false;

  const MachineFunction *MF = MI.getParent()->getParent();
  const TargetLowering *TLI = MF->getSubtarget().getTargetLowering();
  unsigned SP = TLI->getStackPointerRegisterToSaveRestore();
  Register FP = TRI->getFrameRegister(*MF);
  Register Reg = Op.getReg();

  return Reg && Reg != SP && Reg != FP;
}

namespace {

using DefinedRegsSet = SmallSet<Register, 32>;
using VarLocSet = CoalescingBitVector<uint64_t>;

/// A type-checked pair of {Register Location (or 0), Index}, used to index
/// into a \ref VarLocMap. This can be efficiently converted to a 64-bit int
/// for insertion into a \ref VarLocSet, and efficiently converted back. The
/// type-checker helps ensure that the conversions aren't lossy.
///
/// Why encode a location /into/ the VarLocMap index? This makes it possible
/// to find the open VarLocs killed by a register def very quickly. This is a
/// performance-critical operation for LiveDebugValues.
///
/// TODO: Consider adding reserved intervals for kinds of VarLocs other than
/// RegisterKind, like SpillLocKind or EntryValueKind, to optimize iteration
/// over open locations.
struct LocIndex {
  uint32_t Location; // Physical registers live in the range [1;2^30) (see
                     // \ref MCRegister), so we have plenty of range left here
                     // to encode non-register locations.
  uint32_t Index;

  LocIndex(uint32_t Location, uint32_t Index)
      : Location(Location), Index(Index) {}

  uint64_t getAsRawInteger() const {
    return (static_cast<uint64_t>(Location) << 32) | Index;
  }

  template<typename IntT> static LocIndex fromRawInteger(IntT ID) {
    static_assert(std::is_unsigned<IntT>::value &&
                      sizeof(ID) == sizeof(uint64_t),
                  "Cannot convert raw integer to LocIndex");
    return {static_cast<uint32_t>(ID >> 32), static_cast<uint32_t>(ID)};
  }

  /// Get the start of the interval reserved for VarLocs of kind RegisterKind
  /// which reside in \p Reg. The end is at rawIndexForReg(Reg+1)-1.
  static uint64_t rawIndexForReg(uint32_t Reg) {
    return LocIndex(Reg, 0).getAsRawInteger();
  }
};

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

class ValueIDNum {
public:
  uint64_t BlockNo : 16;
  uint64_t InstNo : 20;
  uint64_t LocNo : 14;

  uint64_t asU64() const {
    uint64_t tmp_block = BlockNo;
    uint64_t tmp_inst = InstNo;
    return tmp_block << 34ull | tmp_inst << 14 | LocNo;
  }

  static ValueIDNum fromU64(uint64_t v) {
    return {v >> 34ull, ((v >> 14) & 0xFFFFF), v & 0x3FFF};
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

  std::string asString(const TargetRegisterInfo *TRI) const {
    std::string regname;
    if (LocNo < TRI->getNumRegs())
      regname = TRI->getRegAsmName(LocNo).str();
    else
      regname = Twine("slot ").concat(Twine(LocNo - TRI->getNumRegs())).str();
    assert(regname != "");
    return Twine("bb ").concat(
           Twine(BlockNo).concat(
           Twine(" inst ").concat(
           Twine(InstNo).concat(
           Twine(" loc ").concat(
           Twine(regname)))))).str();
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
} // end namespace llvm


namespace {

class VarLocPos {
public:
  ValueIDNum ID;
  uint64_t CurrentLoc : 14;

  uint64_t asU64() const {
    return ID.asU64() << 14 | CurrentLoc;
  }

  static VarLocPos fromU64(uint64_t v) {
    return {ValueIDNum::fromU64(v >> 14), v & 0x3FFF};
  }

  bool operator==(const VarLocPos &Other) const {
    return std::tie(ID, CurrentLoc) == std::tie(Other.ID, Other.CurrentLoc);
  }

  std::string asString(const TargetRegisterInfo *TRI) const {
    std::string regname;
    if (CurrentLoc < TRI->getNumRegs())
      regname = TRI->getRegAsmName(CurrentLoc).str();
    else
      regname = Twine("slot ").concat(Twine(CurrentLoc - TRI->getNumRegs())).str();
    return Twine("VLP(").concat(ID.asString(TRI)).concat(",cur ").concat(regname).concat(")").str();
  }
};

typedef std::pair<const DIExpression *, bool> MetaVal;

class ValueRec {
public:
  ValueIDNum ID;
  Optional<MachineOperand> MO;
  MetaVal meta;
  unsigned BlockPHI = 0;

  typedef enum { Def, Const, PHI } KindT;
  KindT Kind;

  void dump(const TargetRegisterInfo *TRI) const {
    if (Kind == Const) {
      MO->dump();
    } else if (Kind == PHI) {
      dbgs() << "PHI-bb" << BlockPHI << "\n";
    } else {
      assert(Kind == Def);
      dbgs() << ID.asString(TRI);
    }
    if (meta.second)
      dbgs() << " indir";
    if (meta.first)
      dbgs() << " " << *meta.first;
  }

  bool operator<(const ValueRec &Other) const {
    if (meta != Other.meta)
      return meta < Other.meta;

    if (Kind == Const && Other.Kind == Const) {
      if (MO->getType() == Other.MO->getType()) {
        if (MO->isImm())
          return MO->getImm() < Other.MO->getImm(); 
        else if (MO->isCImm())
          return MO->getCImm() < Other.MO->getCImm(); 
        else if (MO->isFPImm())
          return MO->getFPImm() < Other.MO->getFPImm(); 
        else
          abort();
      } else {
        return MO->getType() < Other.MO->getType();
      }
    } else if (Kind == PHI && Other.Kind == PHI) {
      return BlockPHI < Other.BlockPHI;
    } else if (Kind == Def && Other.Kind == Def) {
      return ID < Other.ID;
    } else {
      return Kind < Other.Kind;
    }
  }
};

typedef UniqueVector<std::pair<DebugVariable, ValueRec>> lolnumberingt;
typedef DenseMap<uint64_t, uint64_t> vphitomphit;
typedef DenseMap<std::pair<const MachineBasicBlock *, ValueIDNum>, ValueIDNum> mphiremapt;

class MLocTracker {
public:
  VarLocSet::Allocator &Alloc;
  std::vector<ValueIDNum> MachineLocsToIDNums;
  unsigned NumRegs;
  UniqueVector<SpillLoc> SpillsToMLocs;

  MLocTracker(VarLocSet::Allocator &Alloc, unsigned NumRegs)
    : Alloc(Alloc), NumRegs(NumRegs) {
    MachineLocsToIDNums.resize(NumRegs);
    reset();
  }

  VarLocPos getVarLocPos(unsigned idx) const {
    return {MachineLocsToIDNums[idx], idx};
  }

  unsigned getNumLocs(void) const {
    return MachineLocsToIDNums.size();
  }

  VarLocSet makeVarLocSet(void) const {
    VarLocSet set(Alloc);
    for (unsigned idx = 0; idx < MachineLocsToIDNums.size(); ++idx) {
      if (MachineLocsToIDNums[idx].LocNo == 0)
        continue;
      set.set(getVarLocPos(idx).asU64());
    }
    return set;
  }

  void loadFromVarLocSet(const VarLocSet &vls, unsigned cur_bb) {
    // Quickly reset everything to being itself at inst 0, representing a phi.
    for (unsigned ID = 0; ID < MachineLocsToIDNums.size(); ++ID) {
      MachineLocsToIDNums[ID] = {cur_bb, 0, ID};
    }

    for (auto ID : vls) {
      auto pos = VarLocPos::fromU64(ID);
      MachineLocsToIDNums[pos.CurrentLoc] = pos.ID;
    }
  }

  void lolremap(const MachineBasicBlock *MBB, const mphiremapt &mphiremap) {
    for (unsigned ID = 0; ID < MachineLocsToIDNums.size(); ++ID) {
      if (MachineLocsToIDNums[ID].InstNo == 0) {
        auto it = mphiremap.find(std::make_pair(MBB, MachineLocsToIDNums[ID]));
        if (it != mphiremap.end())
          MachineLocsToIDNums[ID] = it->second;
      }
    }
  }

  void reset(void) {
    memset(&MachineLocsToIDNums[0], 0, MachineLocsToIDNums.size() * sizeof(ValueIDNum));
  }

  void clear(void) {
    MachineLocsToIDNums.clear();
    //SpillsToMLocs.reset(); XXX can't reset?
    SpillsToMLocs = decltype(SpillsToMLocs)();
  }

  void defReg(Register r, unsigned bb, unsigned inst) {
    ValueIDNum id = {bb, inst, r};
    MachineLocsToIDNums[r] = id;
  }

  void setReg(Register r, ValueIDNum id) {
    MachineLocsToIDNums[r] = id;
  }

  // Because we need to replicate values only having one location for now.
  void lolwipe(Register r) {
    MachineLocsToIDNums[r] = {0, 0, 0};
  }

  void defSpill(SpillLoc l, unsigned bb, unsigned inst) {
    unsigned SpillID = SpillsToMLocs.idFor(l);
    if (SpillID == 0) {
      SpillID = SpillsToMLocs.insert(l);
      SpillID += NumRegs - 1;
      ValueIDNum id = {bb, inst, SpillID};
      MachineLocsToIDNums.push_back(id);
      assert(MachineLocsToIDNums.size() == SpillID + 1);
    } else {
      ValueIDNum id = {bb, inst, SpillID + NumRegs - 1};
      MachineLocsToIDNums[NumRegs + SpillID - 1] = id;
    }
  }

  // xxx duplication
  void setSpill(SpillLoc l, ValueIDNum id) {
    unsigned SpillID = SpillsToMLocs.idFor(l);
    if (SpillID == 0) {
      SpillID = SpillsToMLocs.insert(l);
      SpillID += NumRegs - 1;
      MachineLocsToIDNums.push_back(id);
      assert(MachineLocsToIDNums.size() == SpillID + 1);
    } else {
      MachineLocsToIDNums[NumRegs + SpillID - 1] = id;
    }
  }

  void lolwipe(SpillLoc l) {
    unsigned SpillID = SpillsToMLocs.idFor(l);
    assert(SpillID != 0);
    MachineLocsToIDNums[NumRegs + SpillID - 1] = {0, 0, 0};
  }



  ValueIDNum readReg(Register r) {
    return MachineLocsToIDNums[r];
  }

  ValueIDNum readSpill(SpillLoc l) {
    unsigned pos = SpillsToMLocs.idFor(l);
    if (pos == 0)
      // Returning no location -> 0 means $noreg and some hand wavey position
      return {0, 0, 0};
    return MachineLocsToIDNums[NumRegs + pos - 1];
  }

  unsigned getSpillMLoc(SpillLoc l) {
    unsigned SpillID = SpillsToMLocs.idFor(l);
    if (SpillID == 0)
      return 0;
    SpillID += NumRegs - 1;
    return SpillID;
  }

  bool isSpill(unsigned mloc) const {
    return mloc >= NumRegs;
  }

  void dump(const TargetRegisterInfo *TRI) {
    for (unsigned int ID = 0; ID < NumRegs; ++ID) {
      auto &num = MachineLocsToIDNums[ID];
      if (num.LocNo == 0)
        continue;
      std::string defname = num.asString(TRI);
      dbgs() << TRI->getRegAsmName(ID) << " --> " << defname << "\n";
    }
    for (unsigned int ID = NumRegs; ID < MachineLocsToIDNums.size(); ++ID) {
      auto &num = MachineLocsToIDNums[ID];
      if (num.LocNo == 0)
        continue;
      std::string lolslot = Twine("slot ").concat(Twine(ID - NumRegs)).str();
      std::string defname = num.asString(TRI);
      dbgs() << lolslot << " --> " << defname << "\n";
    }
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
  // xxx determinism?
  // This is the one that actually reduces things :o
  MapVector<DebugVariable, ValueRec> Vars;

public:
  VLocTracker() {}

  void defVar(const MachineInstr &MI, ValueIDNum ID) {
    // XXX skipping overlapping fragments for now.
    assert(MI.isDebugValue());
    DebugVariable Var(MI.getDebugVariable(), MI.getDebugExpression(),
                      MI.getDebugLoc()->getInlinedAt());
    MetaVal m = {MI.getDebugExpression(), MI.getOperand(1).isImm()};
    Vars[Var] = {ID, None, m, 0, ValueRec::Def};
  }

  void defVar(const MachineInstr &MI, const MachineOperand &MO) {
    // XXX skipping overlapping fragments for now.
    assert(MI.isDebugValue());
    DebugVariable Var(MI.getDebugVariable(), MI.getDebugExpression(),
                      MI.getDebugLoc()->getInlinedAt());
    MetaVal m = {MI.getDebugExpression(), MI.getOperand(1).isImm()};
    Vars[Var] = {{0, 0, 0}, MO, m, 0, ValueRec::Const};
  }
};

class TransferTracker {
public:
  const TargetInstrInfo *TII;
  MLocTracker *mlocs;
  MachineFunction &MF;

  struct Transfer {
    MachineBasicBlock::iterator pos;
    MachineBasicBlock *MBB;
    std::vector<MachineInstr *> insts;
  };

  typedef std::pair<unsigned, MetaVal> hahaloc;
  std::vector<Transfer> Transfers;

  // MapVector for nondeterminism
  DenseMap<unsigned, MapVector<DebugVariable, unsigned>> ActiveMLocs;
  DenseMap<DebugVariable, hahaloc> ActiveVLocs;

  TransferTracker(const TargetInstrInfo *TII, MLocTracker *mlocs, MachineFunction &MF) : TII(TII), mlocs(mlocs), MF(MF) { }

  void loadInlocs(MachineBasicBlock &MBB, lolnumberingt &lolnumbering, const mphiremapt &mphiremap, VarLocSet &mlocs, VarLocSet &vlocs, unsigned cur_bb) {  
    ActiveMLocs.clear();
    ActiveVLocs.clear();

    DenseMap<ValueIDNum, unsigned> tmpmap;

    for (auto ID : mlocs) {
      // Each mloc is a VarLocPos
      auto VLP = VarLocPos::fromU64(ID);
      // Produce a map of value numbers to the current machine locs they live
      // in. There should only be one machine loc per value.
      assert(tmpmap.find(VLP.ID) == tmpmap.end()); // XXX expensie
      tmpmap[VLP.ID] = VLP.CurrentLoc;
    }

    // Now map variables to their current machine locs
    std::vector<MachineInstr *> inlocs;
    for (auto ID : vlocs) {
      auto &Var = lolnumbering[ID];
      if (Var.second.Kind == ValueRec::Const) {
        inlocs.push_back(emitMOLoc(*Var.second.MO, Var.first, Var.second.meta));
        continue;
      }

      // Unresolved PHI -> skip
      if (Var.second.Kind == ValueRec::PHI)
        continue;
      assert(Var.second.Kind == ValueRec::Def);

      auto InsertLiveIn = [&](unsigned m) {
        ActiveVLocs[Var.first] = std::make_pair(m, Var.second.meta);
        ActiveMLocs[m].insert(std::make_pair(Var.first, 0));
        assert(m != 0);
        inlocs.push_back(emitLoc(m, Var.first, Var.second.meta));
      };


      // Value unavailable / has no machine loc -> define no location.
      auto hahait = tmpmap.find(Var.second.ID);
      if (hahait != tmpmap.end()) {
        InsertLiveIn(hahait->second);
        continue;
      }

      // Unless this is actually an mloc phi,
      auto &IDNum = Var.second.ID;
      if (IDNum.InstNo != 0)
        continue;

      // Possssiiibbblly remap it.
      // Complete bullshit code, but just proving a point right now.
      auto mphiit= mphiremap.find(std::make_pair(&MBB, IDNum));
      if (mphiit != mphiremap.end()) {
        auto again = tmpmap.find(mphiit->second);
        if (again != tmpmap.end()) {
          InsertLiveIn(again->second);
        } else if (mphiit->second.BlockNo == cur_bb && mphiit->second.InstNo == 0) {
          InsertLiveIn(mphiit->second.LocNo);
        }
      } else if (IDNum.BlockNo == cur_bb) {
        InsertLiveIn(IDNum.LocNo);
      }
    }
    if (inlocs.size() > 0)
      Transfers.push_back({MBB.begin(), &MBB, std::move(inlocs)});
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

    unsigned Reg = MO.getReg();
    MetaVal meta = {MI.getDebugExpression(), MI.getOperand(1).isImm()};

    ActiveMLocs[Reg].insert(std::make_pair(Var, 0));
    if (It == ActiveVLocs.end()) {
      ActiveVLocs.insert(std::make_pair(Var, std::make_pair(Reg, meta)));
    } else {
      It->second.first = Reg;
      It->second.second = meta;
    }
  }

  void clobberMloc(unsigned mloc, MachineBasicBlock::iterator pos) {
    auto It = ActiveMLocs.find(mloc);
    if (It == ActiveMLocs.end())
      return;

    std::vector<MachineInstr *>insts;
    for (auto &Var : It->second) {
      auto ALoc = ActiveVLocs.find(Var.first);
      if (mlocs->isSpill(mloc)) {
        // Create an undef. We can't feed in a nullptr DIExpression alas,
        // so use the variables last expression.
        const DIExpression *Expr = ALoc->second.second.first;
        insts.push_back(emitLoc(0, Var.first, {Expr, false}));
      }
      ActiveVLocs.erase(ALoc);
    }
    if (insts.size() != 0)
      Transfers.push_back({std::next(pos), pos->getParent(), std::move(insts)});

    It->second.clear();
  }

  void transferMlocs(unsigned src, unsigned dst, MachineBasicBlock::iterator pos) {
    // Legitimate scenario on account of un-clobbered slot being assigned to?
    //assert(ActiveMLocs[dst].size() == 0);
    ActiveMLocs[dst] = ActiveMLocs[src];

    std::vector<MachineInstr *> instrs;
    for (auto &Var : ActiveMLocs[src]) {
      auto it = ActiveVLocs.find(Var.first);
      assert(it != ActiveVLocs.end());
      it->second.first = dst;

      assert(dst != 0);
      MachineInstr *MI = emitLoc(dst, Var.first, it->second.second);
      instrs.push_back(MI);
    }
    ActiveMLocs[src].clear();
    if (instrs.size() > 0)
      Transfers.push_back({std::next(pos), pos->getParent(), std::move(instrs)});
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

  MachineInstrBuilder 
  emitLoc(unsigned MLoc, const DebugVariable &Var, const MetaVal &meta) {
    DebugLoc DL = DebugLoc::get(0, 0, Var.getVariable()->getScope(), Var.getInlinedAt());
    auto MIB = BuildMI(MF, DL, TII->get(TargetOpcode::DBG_VALUE));

    const DIExpression *Expr = meta.first;
    if (MLoc < mlocs->NumRegs) {
      MIB.addReg(MLoc, RegState::Debug);
      if (meta.second)
        MIB.addImm(0);
      else
        MIB.addReg(0, RegState::Debug);
    } else {
      const SpillLoc &Loc = mlocs->SpillsToMLocs[MLoc - mlocs->NumRegs + 1];
      Expr = DIExpression::prepend(Expr, DIExpression::ApplyOffset, Loc.SpillOffset);
      unsigned Base = Loc.SpillBase;
      MIB.addReg(Base, RegState::Debug);
      MIB.addImm(0);
    }

    MIB.addMetadata(Var.getVariable());
    MIB.addMetadata(Expr);
    return MIB;
  }
};

/// Keeps track of lexical scopes associated with a user value's source
/// location.
class UserValueScopes {
  DebugLoc DL;
  LexicalScopes &LS;
  SmallPtrSet<const MachineBasicBlock *, 4> LBlocks;

public:
  UserValueScopes(DebugLoc D, LexicalScopes &L) : DL(std::move(D)), LS(L) {}

  /// Return true if current scope dominates at least one machine
  /// instruction in a given machine basic block.
  bool dominates(MachineBasicBlock *MBB) {
    if (LBlocks.empty())
      LS.getMachineBasicBlocks(DL, LBlocks);
    return LBlocks.count(MBB) != 0 || LS.dominates(DL, MBB);
  }
};

class LiveDebugValues : public MachineFunctionPass {
private:
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;
  const TargetFrameLowering *TFI;
  BitVector CalleeSavedRegs;
  LexicalScopes LS;
  VarLocSet::Allocator Alloc;

  MLocTracker *tracker;
  unsigned cur_bb;
  unsigned cur_inst;
  VLocTracker *vtracker;
  TransferTracker *ttracker;

  enum struct TransferKind { TransferCopy, TransferSpill, TransferRestore };


  using FragmentInfo = DIExpression::FragmentInfo;
  using OptFragmentInfo = Optional<DIExpression::FragmentInfo>;

  /// A pair of debug variable and value location.
  struct VarLoc {
    /// Identity of the variable at this location.
    const DebugVariable Var;

    /// The expression applied to this location.
    const DIExpression *Expr;

    /// DBG_VALUE to clone var/expr information from if this location
    /// is moved.
    const MachineInstr &MI;

    mutable UserValueScopes UVS;
    enum VarLocKind {
      InvalidKind = 0,
      RegisterKind,
      SpillLocKind,
      ImmediateKind,
      EntryValueKind,
      EntryValueBackupKind,
      EntryValueCopyBackupKind
    } Kind = InvalidKind;

    /// The value location. Stored separately to avoid repeatedly
    /// extracting it from MI.
    union {
      uint64_t RegNo;
      SpillLoc SpillLocation;
      uint64_t Hash;
      int64_t Immediate;
      const ConstantFP *FPImm;
      const ConstantInt *CImm;
    } Loc;

    VarLoc(const MachineInstr &MI, LexicalScopes &LS)
        : Var(MI.getDebugVariable(), MI.getDebugExpression(),
              MI.getDebugLoc()->getInlinedAt()),
          Expr(MI.getDebugExpression()), MI(MI), UVS(MI.getDebugLoc(), LS) {
      static_assert((sizeof(Loc) == sizeof(uint64_t)),
                    "hash does not cover all members of Loc");
      assert(MI.isDebugValue() && "not a DBG_VALUE");
      assert(MI.getNumOperands() == 4 && "malformed DBG_VALUE");
      if (int RegNo = isDbgValueDescribedByReg(MI)) {
        Kind = RegisterKind;
        Loc.RegNo = RegNo;
      } else if (MI.getOperand(0).isImm()) {
        Kind = ImmediateKind;
        Loc.Immediate = MI.getOperand(0).getImm();
      } else if (MI.getOperand(0).isFPImm()) {
        Kind = ImmediateKind;
        Loc.FPImm = MI.getOperand(0).getFPImm();
      } else if (MI.getOperand(0).isCImm()) {
        Kind = ImmediateKind;
        Loc.CImm = MI.getOperand(0).getCImm();
      }

      // We create the debug entry values from the factory functions rather than
      // from this ctor.
      assert(Kind != EntryValueKind && !isEntryBackupLoc());
    }

    /// Take the variable and machine-location in DBG_VALUE MI, and build an
    /// entry location using the given expression.
    static VarLoc CreateEntryLoc(const MachineInstr &MI, LexicalScopes &LS,
                                 const DIExpression *EntryExpr, unsigned Reg) {
      VarLoc VL(MI, LS);
      assert(VL.Kind == RegisterKind);
      VL.Kind = EntryValueKind;
      VL.Expr = EntryExpr;
      VL.Loc.RegNo = Reg;
      return VL;
    }

    /// Take the variable and machine-location from the DBG_VALUE (from the
    /// function entry), and build an entry value backup location. The backup
    /// location will turn into the normal location if the backup is valid at
    /// the time of the primary location clobbering.
    static VarLoc CreateEntryBackupLoc(const MachineInstr &MI,
                                       LexicalScopes &LS,
                                       const DIExpression *EntryExpr) {
      VarLoc VL(MI, LS);
      assert(VL.Kind == RegisterKind);
      VL.Kind = EntryValueBackupKind;
      VL.Expr = EntryExpr;
      return VL;
    }

    /// Take the variable and machine-location from the DBG_VALUE (from the
    /// function entry), and build a copy of an entry value backup location by
    /// setting the register location to NewReg.
    static VarLoc CreateEntryCopyBackupLoc(const MachineInstr &MI,
                                           LexicalScopes &LS,
                                           const DIExpression *EntryExpr,
                                           unsigned NewReg) {
      VarLoc VL(MI, LS);
      assert(VL.Kind == RegisterKind);
      VL.Kind = EntryValueCopyBackupKind;
      VL.Expr = EntryExpr;
      VL.Loc.RegNo = NewReg;
      return VL;
    }

    /// Copy the register location in DBG_VALUE MI, updating the register to
    /// be NewReg.
    static VarLoc CreateCopyLoc(const MachineInstr &MI, LexicalScopes &LS,
                                unsigned NewReg) {
      VarLoc VL(MI, LS);
      assert(VL.Kind == RegisterKind);
      VL.Loc.RegNo = NewReg;
      return VL;
    }

    /// Take the variable described by DBG_VALUE MI, and create a VarLoc
    /// locating it in the specified spill location.
    static VarLoc CreateSpillLoc(const MachineInstr &MI, unsigned SpillBase,
                                 int SpillOffset, LexicalScopes &LS) {
      VarLoc VL(MI, LS);
      assert(VL.Kind == RegisterKind);
      VL.Kind = SpillLocKind;
      VL.Loc.SpillLocation = {SpillBase, SpillOffset};
      return VL;
    }

    /// Create a DBG_VALUE representing this VarLoc in the given function.
    /// Copies variable-specific information such as DILocalVariable and
    /// inlining information from the original DBG_VALUE instruction, which may
    /// have been several transfers ago.
    MachineInstr *BuildDbgValue(MachineFunction &MF) const {
      const DebugLoc &DbgLoc = MI.getDebugLoc();
      bool Indirect = MI.isIndirectDebugValue();
      const auto &IID = MI.getDesc();
      const DILocalVariable *Var = MI.getDebugVariable();
      const DIExpression *DIExpr = MI.getDebugExpression();

      switch (Kind) {
      case EntryValueKind:
        // An entry value is a register location -- but with an updated
        // expression. The register location of such DBG_VALUE is always the one
        // from the entry DBG_VALUE, it does not matter if the entry value was
        // copied in to another register due to some optimizations.
        return BuildMI(MF, DbgLoc, IID, Indirect, MI.getOperand(0).getReg(),
                       Var, Expr);
      case RegisterKind:
        // Register locations are like the source DBG_VALUE, but with the
        // register number from this VarLoc.
        return BuildMI(MF, DbgLoc, IID, Indirect, Loc.RegNo, Var, DIExpr);
      case SpillLocKind: {
        // Spills are indirect DBG_VALUEs, with a base register and offset.
        // Use the original DBG_VALUEs expression to build the spilt location
        // on top of. FIXME: spill locations created before this pass runs
        // are not recognized, and not handled here.
        auto *SpillExpr = DIExpression::prepend(
            DIExpr, DIExpression::ApplyOffset, Loc.SpillLocation.SpillOffset);
        unsigned Base = Loc.SpillLocation.SpillBase;
        return BuildMI(MF, DbgLoc, IID, true, Base, Var, SpillExpr);
      }
      case ImmediateKind: {
        MachineOperand MO = MI.getOperand(0);
        return BuildMI(MF, DbgLoc, IID, Indirect, MO, Var, DIExpr);
      }
      case EntryValueBackupKind:
      case EntryValueCopyBackupKind:
      case InvalidKind:
        llvm_unreachable(
            "Tried to produce DBG_VALUE for invalid or backup VarLoc");
      }
      llvm_unreachable("Unrecognized LiveDebugValues.VarLoc.Kind enum");
    }

    /// Is the Loc field a constant or constant object?
    bool isConstant() const { return Kind == ImmediateKind; }

    /// Check if the Loc field is an entry backup location.
    bool isEntryBackupLoc() const {
      return Kind == EntryValueBackupKind || Kind == EntryValueCopyBackupKind;
    }

    /// If this variable is described by a register holding the entry value,
    /// return it, otherwise return 0.
    unsigned getEntryValueBackupReg() const {
      if (Kind == EntryValueBackupKind)
        return Loc.RegNo;
      return 0;
    }

    /// If this variable is described by a register holding the copy of the
    /// entry value, return it, otherwise return 0.
    unsigned getEntryValueCopyBackupReg() const {
      if (Kind == EntryValueCopyBackupKind)
        return Loc.RegNo;
      return 0;
    }

    /// If this variable is described by a register, return it,
    /// otherwise return 0.
    unsigned isDescribedByReg() const {
      if (Kind == RegisterKind)
        return Loc.RegNo;
      return 0;
    }

    /// Determine whether the lexical scope of this value's debug location
    /// dominates MBB.
    bool dominates(MachineBasicBlock &MBB) const { return UVS.dominates(&MBB); }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    // TRI can be null.
    void dump(const TargetRegisterInfo *TRI, raw_ostream &Out = dbgs()) const {
      dbgs() << "VarLoc(";
      switch (Kind) {
      case RegisterKind:
      case EntryValueKind:
      case EntryValueBackupKind:
      case EntryValueCopyBackupKind:
        dbgs() << printReg(Loc.RegNo, TRI);
        break;
      case SpillLocKind:
        dbgs() << printReg(Loc.SpillLocation.SpillBase, TRI);
        dbgs() << "[" << Loc.SpillLocation.SpillOffset << "]";
        break;
      case ImmediateKind:
        dbgs() << Loc.Immediate;
        break;
      case InvalidKind:
        llvm_unreachable("Invalid VarLoc in dump method");
      }

      dbgs() << ", \"" << Var.getVariable()->getName() << "\", " << *Expr
             << ", ";
      if (Var.getInlinedAt())
        dbgs() << "!" << Var.getInlinedAt()->getMetadataID() << ")\n";
      else
        dbgs() << "(null))";

      if (isEntryBackupLoc())
        dbgs() << " (backup loc)\n";
      else
        dbgs() << "\n";
    }
#endif

    bool operator==(const VarLoc &Other) const {
      return Kind == Other.Kind && Var == Other.Var &&
             Loc.Hash == Other.Loc.Hash && Expr == Other.Expr;
    }

    /// This operator guarantees that VarLocs are sorted by Variable first.
    bool operator<(const VarLoc &Other) const {
      return std::tie(Var, Kind, Loc.Hash, Expr) <
             std::tie(Other.Var, Other.Kind, Other.Loc.Hash, Other.Expr);
    }
  };

  /// VarLocMap is used for two things:
  /// 1) Assigning a unique LocIndex to a VarLoc. This LocIndex can be used to
  ///    virtually insert a VarLoc into a VarLocSet.
  /// 2) Given a LocIndex, look up the unique associated VarLoc.
  class VarLocMap {
    /// Map a VarLoc to an index within the vector reserved for its location
    /// within Loc2Vars.
    std::map<VarLoc, uint32_t> Var2Index;

    /// Map a location to a vector which holds VarLocs which live in that
    /// location.
    SmallDenseMap<uint32_t, std::vector<VarLoc>> Loc2Vars;

  public:
    /// Retrieve a unique LocIndex for \p VL.
    LocIndex insert(const VarLoc &VL) {
      uint32_t Location = VL.isDescribedByReg();
      uint32_t &Index = Var2Index[VL];
      if (!Index) {
        auto &Vars = Loc2Vars[Location];
        Vars.push_back(VL);
        Index = Vars.size();
      }
      return {Location, Index - 1};
    }

    /// Retrieve the unique VarLoc associated with \p ID.
    const VarLoc &operator[](LocIndex ID) const {
      auto LocIt = Loc2Vars.find(ID.Location);
      assert(LocIt != Loc2Vars.end() && "Location not tracked");
      return LocIt->second[ID.Index];
    }
  };

  using VarLocInMBB = SmallDenseMap<const MachineBasicBlock *, VarLocSet>;
  struct TransferDebugPair {
    MachineInstr *TransferInst; ///< Instruction where this transfer occurs.
    LocIndex LocationID;        ///< Location number for the transfer dest.
  };
  using TransferMap = SmallVector<TransferDebugPair, 4>;

  // Helper while building OverlapMap, a map of all fragments seen for a given
  // DILocalVariable.
  using VarToFragments =
      DenseMap<const DILocalVariable *, SmallSet<FragmentInfo, 4>>;

  /// This holds the working set of currently open ranges. For fast
  /// access, this is done both as a set of VarLocIDs, and a map of
  /// DebugVariable to recent VarLocID. Note that a DBG_VALUE ends all
  /// previous open ranges for the same variable. In addition, we keep
  /// two different maps (Vars/EntryValuesBackupVars), so erase/insert
  /// methods act differently depending on whether a VarLoc is primary
  /// location or backup one. In the case the VarLoc is backup location
  /// we will erase/insert from the EntryValuesBackupVars map, otherwise
  /// we perform the operation on the Vars.
  class OpenRangesSet {
    VarLocSet VarLocs;
    // Map the DebugVariable to recent primary location ID.
    SmallDenseMap<DebugVariable, LocIndex, 8> Vars;
    // Map the DebugVariable to recent backup location ID.
    SmallDenseMap<DebugVariable, LocIndex, 8> EntryValuesBackupVars;
    OverlapMap &OverlappingFragments;

  public:
    OpenRangesSet(VarLocSet::Allocator &Alloc, OverlapMap &_OLapMap)
        : VarLocs(Alloc), OverlappingFragments(_OLapMap) {}

    const VarLocSet &getVarLocs() const { return VarLocs; }

    /// Terminate all open ranges for VL.Var by removing it from the set.
    void erase(const VarLoc &VL);

    /// Terminate all open ranges listed in \c KillSet by removing
    /// them from the set.
    void erase(const VarLocSet &KillSet, const VarLocMap &VarLocIDs);

    /// Insert a new range into the set.
    void insert(LocIndex VarLocID, const VarLoc &VL);

    /// Insert a set of ranges.
    void insertFromLocSet(const VarLocSet &ToLoad, const VarLocMap &Map) {
      for (uint64_t ID : ToLoad) {
        LocIndex Idx = LocIndex::fromRawInteger(ID);
        const VarLoc &VarL = Map[Idx];
        insert(Idx, VarL);
      }
    }

    llvm::Optional<LocIndex> getEntryValueBackup(DebugVariable Var);

    /// Empty the set.
    void clear() {
      VarLocs.clear();
      Vars.clear();
      EntryValuesBackupVars.clear();
    }

    /// Return whether the set is empty or not.
    bool empty() const {
      assert(Vars.empty() == EntryValuesBackupVars.empty() &&
             Vars.empty() == VarLocs.empty() &&
             "open ranges are inconsistent");
      return VarLocs.empty();
    }
  };

  /// Collect all VarLoc IDs from \p CollectFrom for VarLocs which are located
  /// in \p Reg, of kind RegisterKind. Insert collected IDs in \p Collected.
  void collectIDsForReg(VarLocSet &Collected, uint32_t Reg,
                        const VarLocSet &CollectFrom) const;

  /// Get the registers which are used by VarLocs of kind RegisterKind tracked
  /// by \p CollectFrom.
  void getUsedRegs(const VarLocSet &CollectFrom,
                   SmallVectorImpl<uint32_t> &UsedRegs) const;

  VarLocSet &getVarLocsInMBB(const MachineBasicBlock *MBB, VarLocInMBB &Locs) {
    auto Result = Locs.try_emplace(MBB, Alloc);
    return Result.first->second;
  }

  const VarLocSet &getVarLocsInMBB(const MachineBasicBlock *MBB,
                                   const VarLocInMBB &Locs) const {
    auto It = Locs.find(MBB);
    assert(It != Locs.end() && "MBB not in map");
    return It->second;
  }

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

  /// Returns true if the given machine instruction is a debug value which we
  /// can emit entry values for.
  ///
  /// Currently, we generate debug entry values only for parameters that are
  /// unmodified throughout the function and located in a register.
  bool isEntryValueCandidate(const MachineInstr &MI,
                             const DefinedRegsSet &Regs) const;

  /// If a given instruction is identified as a spill, return the spill location
  /// and set \p Reg to the spilled register.
  Optional<SpillLoc> isRestoreInstruction(const MachineInstr &MI,
                                                  MachineFunction *MF,
                                                  unsigned &Reg);
  /// Given a spill instruction, extract the register and offset used to
  /// address the spill location in a target independent way.
  SpillLoc extractSpillBaseRegAndOffset(const MachineInstr &MI);
  void insertTransferDebugPair(MachineInstr &MI, OpenRangesSet &OpenRanges,
                               TransferMap *Transfers, VarLocMap &VarLocIDs,
                               LocIndex OldVarID, TransferKind Kind,
                               unsigned NewReg = 0);

  void transferDebugValue(const MachineInstr &MI, OpenRangesSet &OpenRanges,
                          VarLocMap &VarLocIDs);
  void transferSpillOrRestoreInst(MachineInstr &MI, OpenRangesSet &OpenRanges,
                                  VarLocMap &VarLocIDs, TransferMap *Transfers);
  bool removeEntryValue(const MachineInstr &MI, OpenRangesSet &OpenRanges,
                        VarLocMap &VarLocIDs, const VarLoc &EntryVL);
  void emitEntryValues(MachineInstr &MI, OpenRangesSet &OpenRanges,
                       VarLocMap &VarLocIDs, TransferMap *Transfers,
                       VarLocSet &KillSet);
  void recordEntryValue(const MachineInstr &MI,
                        const DefinedRegsSet &DefinedRegs,
                        OpenRangesSet &OpenRanges, VarLocMap &VarLocIDs);
  void transferRegisterCopy(MachineInstr &MI, OpenRangesSet &OpenRanges,
                            VarLocMap &VarLocIDs, TransferMap *Transfers);
  void transferRegisterDef(MachineInstr &MI, OpenRangesSet &OpenRanges,
                           VarLocMap &VarLocIDs, TransferMap *Transfers);
  bool transferTerminator(MachineBasicBlock *MBB, OpenRangesSet &OpenRanges,
                          VarLocInMBB &OutLocs, const VarLocMap &VarLocIDs);

  void process(MachineInstr &MI, OpenRangesSet &OpenRanges,
               VarLocMap &VarLocIDs, TransferMap *Transfers);

  void accumulateFragmentMap(MachineInstr &MI, VarToFragments &SeenFragments,
                             OverlapMap &OLapMap);

  bool join(MachineBasicBlock &MBB, VarLocInMBB &OutLocs, VarLocInMBB &InLocs,
            const VarLocMap &VarLocIDs,
            SmallPtrSet<const MachineBasicBlock *, 16> &Visited,
            SmallPtrSetImpl<const MachineBasicBlock *> &ArtificialBlocks,
            VarLocInMBB &PendingInLocs, bool mlocs);

  bool vloc_join(const MachineBasicBlock &MBB, VarLocInMBB &VLOCOutLocs,
                 VarLocInMBB &VLOCInLocs, lolnumberingt &lolnumbering,
                 SmallPtrSet<const MachineBasicBlock *, 16> &VLOCVisited,
                 SmallPtrSetImpl<const MachineBasicBlock *> &ArtificialBlocks,
                 VarLocInMBB &VLOCPendingInLocs, unsigned cur_bb);
  bool vloc_transfer(VarLocSet &ilocs, VarLocSet &transfer, VarLocSet &olocs, lolnumberingt &lolnumbering);


  void resolveMPHIs(mphiremapt &mphiremap, MachineBasicBlock &MBB, VarLocSet &InLocs, VarLocInMBB &MLOCOutLocs, unsigned cur_bb);
  void resolveVPHIs(vphitomphit &vphitomphi, const mphiremapt &mphiremap, lolnumberingt &lolnumbering, MachineBasicBlock &MBB, VarLocSet &InLocs, VarLocInMBB &VLOCOutLocs, VarLocInMBB &MLOCOutLocs, unsigned cur_bb);

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

  /// Print to ostream with a message.
  void printVarLocInMBB(const MachineFunction &MF, const VarLocInMBB &V,
                        const VarLocMap &VarLocIDs, const char *msg,
                        raw_ostream &Out) const;

  void emitDbgValue(MLocTracker *mlocs, ValueRec &theloc, DebugVariable &Var,
                    MachineBasicBlock::iterator pos);

  /// Calculate the liveness information for the given machine function.
  bool runOnMachineFunction(MachineFunction &MF) override;
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

/// Erase a variable from the set of open ranges, and additionally erase any
/// fragments that may overlap it. If the VarLoc is a buckup location, erase
/// the variable from the EntryValuesBackupVars set, indicating we should stop
/// tracking its backup entry location. Otherwise, if the VarLoc is primary
/// location, erase the variable from the Vars set.
void LiveDebugValues::OpenRangesSet::erase(const VarLoc &VL) {
  return;
}

void LiveDebugValues::OpenRangesSet::erase(const VarLocSet &KillSet,
                                           const VarLocMap &VarLocIDs) {
  return;
}

void LiveDebugValues::OpenRangesSet::insert(LocIndex VarLocID,
                                            const VarLoc &VL) {
  return;
}

/// Return the Loc ID of an entry value backup location, if it exists for the
/// variable.
llvm::Optional<LocIndex>
LiveDebugValues::OpenRangesSet::getEntryValueBackup(DebugVariable Var) {
  auto It = EntryValuesBackupVars.find(Var);
  if (It != EntryValuesBackupVars.end())
    return It->second;

  return llvm::None;
}

void LiveDebugValues::collectIDsForReg(VarLocSet &Collected, uint32_t Reg,
                                       const VarLocSet &CollectFrom) const {
  // The half-open interval [FirstIndexForReg, FirstInvalidIndex) contains all
  // possible VarLoc IDs for VarLocs of kind RegisterKind which live in Reg.
  uint64_t FirstIndexForReg = LocIndex::rawIndexForReg(Reg);
  uint64_t FirstInvalidIndex = LocIndex::rawIndexForReg(Reg + 1);
  // Iterate through that half-open interval and collect all the set IDs.
  for (auto It = CollectFrom.find(FirstIndexForReg), End = CollectFrom.end();
       It != End && *It < FirstInvalidIndex; ++It)
    Collected.set(*It);
}

void LiveDebugValues::getUsedRegs(const VarLocSet &CollectFrom,
                                  SmallVectorImpl<uint32_t> &UsedRegs) const {
  // All register-based VarLocs are assigned indices greater than or equal to
  // FirstRegIndex.
  uint64_t FirstRegIndex = LocIndex::rawIndexForReg(1);
  for (auto It = CollectFrom.find(FirstRegIndex), End = CollectFrom.end();
       It != End;) {
    // We found a VarLoc ID for a VarLoc that lives in a register. Figure out
    // which register and add it to UsedRegs.
    uint32_t FoundReg = LocIndex::fromRawInteger(*It).Location;
    assert((UsedRegs.empty() || FoundReg != UsedRegs.back()) &&
           "Duplicate used reg");
    UsedRegs.push_back(FoundReg);

    // Skip to the next /set/ register. Note that this finds a lower bound, so
    // even if there aren't any VarLocs living in `FoundReg+1`, we're still
    // guaranteed to move on to the next register (or to end()).
    uint64_t NextRegIndex = LocIndex::rawIndexForReg(FoundReg + 1);
    It = CollectFrom.find(NextRegIndex);
  }
}

//===----------------------------------------------------------------------===//
//            Debug Range Extension Implementation
//===----------------------------------------------------------------------===//

#ifndef NDEBUG
void LiveDebugValues::printVarLocInMBB(const MachineFunction &MF,
                                       const VarLocInMBB &V,
                                       const VarLocMap &VarLocIDs,
                                       const char *msg,
                                       raw_ostream &Out) const {
  Out << '\n' << msg << '\n';
  for (const MachineBasicBlock &BB : MF) {
    if (!V.count(&BB))
      continue;
    const VarLocSet &L = getVarLocsInMBB(&BB, V);
    if (L.empty())
      continue;
    Out << "MBB: " << BB.getNumber() << ":\n";
    for (uint64_t VLL : L) {
      const VarLoc &VL = VarLocIDs[LocIndex::fromRawInteger(VLL)];
      Out << " Var: " << VL.Var.getVariable()->getName();
      Out << " MI: ";
      VL.dump(TRI, Out);
    }
  }
  Out << "\n";
}
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

/// Try to salvage the debug entry value if we encounter a new debug value
/// describing the same parameter, otherwise stop tracking the value. Return
/// true if we should stop tracking the entry value, otherwise return false.
bool LiveDebugValues::removeEntryValue(const MachineInstr &MI,
                                       OpenRangesSet &OpenRanges,
                                       VarLocMap &VarLocIDs,
                                       const VarLoc &EntryVL) {
  // Skip the DBG_VALUE which is the debug entry value itself.
  if (MI.isIdenticalTo(EntryVL.MI))
    return false;

  // If the parameter's location is not register location, we can not track
  // the entry value any more. In addition, if the debug expression from the
  // DBG_VALUE is not empty, we can assume the parameter's value has changed
  // indicating that we should stop tracking its entry value as well.
  if (!MI.getOperand(0).isReg() ||
      MI.getDebugExpression()->getNumElements() != 0)
    return true;

  // If the DBG_VALUE comes from a copy instruction that copies the entry value,
  // it means the parameter's value has not changed and we should be able to use
  // its entry value.
  bool TrySalvageEntryValue = false;
  Register Reg = MI.getOperand(0).getReg();
  auto I = std::next(MI.getReverseIterator());
  const MachineOperand *SrcRegOp, *DestRegOp;
  if (I != MI.getParent()->rend()) {
    // TODO: Try to keep tracking of an entry value if we encounter a propagated
    // DBG_VALUE describing the copy of the entry value. (Propagated entry value
    // does not indicate the parameter modification.)
    auto DestSrc = TII->isCopyInstr(*I);
    if (!DestSrc)
      return true;

    SrcRegOp = DestSrc->Source;
    DestRegOp = DestSrc->Destination;
    if (Reg != DestRegOp->getReg())
      return true;
    TrySalvageEntryValue = true;
  }

  if (TrySalvageEntryValue) {
    for (uint64_t ID : OpenRanges.getVarLocs()) {
      const VarLoc &VL = VarLocIDs[LocIndex::fromRawInteger(ID)];
      if (!VL.isEntryBackupLoc())
        continue;

      if (VL.getEntryValueCopyBackupReg() == Reg &&
          VL.MI.getOperand(0).getReg() == SrcRegOp->getReg())
        return false;
    }
  }

  return true;
}

/// End all previous ranges related to @MI and start a new range from @MI
/// if it is a DBG_VALUE instr.
void LiveDebugValues::transferDebugValue(const MachineInstr &MI,
                                         OpenRangesSet &OpenRanges,
                                         VarLocMap &VarLocIDs) {
  if (!MI.isDebugValue())
    return;
  const DILocalVariable *Var = MI.getDebugVariable();
  const DIExpression *Expr = MI.getDebugExpression();
  const DILocation *DebugLoc = MI.getDebugLoc();
  const DILocation *InlinedAt = DebugLoc->getInlinedAt();
  assert(Var->isValidLocationForIntrinsic(DebugLoc) &&
         "Expected inlined-at fields to agree");

  DebugVariable V(Var, Expr, InlinedAt);

  // Check if this DBG_VALUE indicates a parameter's value changing.
  // If that is the case, we should stop tracking its entry value.
  auto EntryValBackupID = OpenRanges.getEntryValueBackup(V);
  if (Var->isParameter() && EntryValBackupID) {
    const VarLoc &EntryVL = VarLocIDs[*EntryValBackupID];
    if (removeEntryValue(MI, OpenRanges, VarLocIDs, EntryVL)) {
      LLVM_DEBUG(dbgs() << "Deleting a DBG entry value because of: ";
                 MI.print(dbgs(), /*IsStandalone*/ false,
                          /*SkipOpers*/ false, /*SkipDebugLoc*/ false,
                          /*AddNewLine*/ true, TII));
      OpenRanges.erase(EntryVL);
    }
  }

  if (isDbgValueDescribedByReg(MI) || MI.getOperand(0).isImm() ||
      MI.getOperand(0).isFPImm() || MI.getOperand(0).isCImm()) {
    // Use normal VarLoc constructor for registers and immediates.
    VarLoc VL(MI, LS);
    // End all previous ranges of VL.Var.
    OpenRanges.erase(VL);

    LocIndex ID = VarLocIDs.insert(VL);
    // Add the VarLoc to OpenRanges from this DBG_VALUE.
    OpenRanges.insert(ID, VL);
  } else if (MI.hasOneMemOperand()) {
    llvm_unreachable("DBG_VALUE with mem operand encountered after regalloc?");
  } else {
    // This must be an undefined location. We should leave OpenRanges closed.
    assert(MI.getOperand(0).isReg() && MI.getOperand(0).getReg() == 0 &&
           "Unexpected non-undef DBG_VALUE encountered");
  }

  if (vtracker) {
    if (isDbgValueDescribedByReg(MI)) {
      auto ID = tracker->readReg(MI.getOperand(0).getReg());
      vtracker->defVar(MI, ID);
    } else if (MI.getOperand(0).isImm() || MI.getOperand(0).isFPImm() ||
               MI.getOperand(0).isCImm()) {
      vtracker->defVar(MI, MI.getOperand(0));
    }
  }

  if (ttracker)
    ttracker->redefVar(MI);
}

/// Turn the entry value backup locations into primary locations.
void LiveDebugValues::emitEntryValues(MachineInstr &MI,
                                      OpenRangesSet &OpenRanges,
                                      VarLocMap &VarLocIDs,
                                      TransferMap *Transfers,
                                      VarLocSet &KillSet) {
  for (uint64_t ID : KillSet) {
    LocIndex Idx = LocIndex::fromRawInteger(ID);
    const VarLoc &VL = VarLocIDs[Idx];
    if (!VL.Var.getVariable()->isParameter())
      continue;

    auto DebugVar = VL.Var;
    Optional<LocIndex> EntryValBackupID =
        OpenRanges.getEntryValueBackup(DebugVar);

    // If the parameter has the entry value backup, it means we should
    // be able to use its entry value.
    if (!EntryValBackupID)
      continue;

    const VarLoc &EntryVL = VarLocIDs[*EntryValBackupID];
    VarLoc EntryLoc =
        VarLoc::CreateEntryLoc(EntryVL.MI, LS, EntryVL.Expr, EntryVL.Loc.RegNo);
    LocIndex EntryValueID = VarLocIDs.insert(EntryLoc);
    OpenRanges.insert(EntryValueID, EntryLoc);

    if (Transfers)
      Transfers->push_back({&MI, EntryValueID});
  }
}

/// Create new TransferDebugPair and insert it in \p Transfers. The VarLoc
/// with \p OldVarID should be deleted form \p OpenRanges and replaced with
/// new VarLoc. If \p NewReg is different than default zero value then the
/// new location will be register location created by the copy like instruction,
/// otherwise it is variable's location on the stack.
void LiveDebugValues::insertTransferDebugPair(
    MachineInstr &MI, OpenRangesSet &OpenRanges, TransferMap *Transfers,
    VarLocMap &VarLocIDs, LocIndex OldVarID, TransferKind Kind,
    unsigned NewReg) {
  const MachineInstr *DebugInstr = &VarLocIDs[OldVarID].MI;

  auto ProcessVarLoc = [&MI, &OpenRanges, &Transfers, &VarLocIDs](VarLoc &VL) {
    LocIndex LocId = VarLocIDs.insert(VL);

    // Close this variable's previous location range.
    OpenRanges.erase(VL);

    // Record the new location as an open range, and a postponed transfer
    // inserting a DBG_VALUE for this location.
    OpenRanges.insert(LocId, VL);
    if (Transfers) {
      TransferDebugPair MIP = {&MI, LocId};
      Transfers->push_back(MIP);
    }
  };

  // End all previous ranges of VL.Var.
  OpenRanges.erase(VarLocIDs[OldVarID]);
  switch (Kind) {
  case TransferKind::TransferCopy: {
    assert(NewReg &&
           "No register supplied when handling a copy of a debug value");
    // Create a DBG_VALUE instruction to describe the Var in its new
    // register location.
    VarLoc VL = VarLoc::CreateCopyLoc(*DebugInstr, LS, NewReg);
    ProcessVarLoc(VL);
    LLVM_DEBUG({
      dbgs() << "Creating VarLoc for register copy:";
      VL.dump(TRI);
    });
    return;
  }
  case TransferKind::TransferSpill: {
    // Create a DBG_VALUE instruction to describe the Var in its spilled
    // location.
    SpillLoc SpillLocation = extractSpillBaseRegAndOffset(MI);
    VarLoc VL = VarLoc::CreateSpillLoc(*DebugInstr, SpillLocation.SpillBase,
                                       SpillLocation.SpillOffset, LS);
    ProcessVarLoc(VL);
    LLVM_DEBUG({
      dbgs() << "Creating VarLoc for spill:";
      VL.dump(TRI);
    });
    return;
  }
  case TransferKind::TransferRestore: {
    assert(NewReg &&
           "No register supplied when handling a restore of a debug value");
    // DebugInstr refers to the pre-spill location, therefore we can reuse
    // its expression.
    VarLoc VL = VarLoc::CreateCopyLoc(*DebugInstr, LS, NewReg);
    ProcessVarLoc(VL);
    LLVM_DEBUG({
      dbgs() << "Creating VarLoc for restore:";
      VL.dump(TRI);
    });
    return;
  }
  }
  llvm_unreachable("Invalid transfer kind");
}

/// A definition of a register may mark the end of a range.
void LiveDebugValues::transferRegisterDef(
    MachineInstr &MI, OpenRangesSet &OpenRanges, VarLocMap &VarLocIDs,
    TransferMap *Transfers) {

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
    }
  }

  // Erase VarLocs which reside in one of the dead registers. For performance
  // reasons, it's critical to not iterate over the full set of open VarLocs.
  // Iterate over the set of dying/used regs instead.
  VarLocSet KillSet(Alloc);
  for (uint32_t DeadReg : DeadRegs) {
    collectIDsForReg(KillSet, DeadReg, OpenRanges.getVarLocs());
    tracker->defReg(DeadReg, cur_bb, cur_inst);
    if (ttracker)
      ttracker->clobberMloc(DeadReg, MI.getIterator());
  }

  auto AnyRegMaskKillsReg = [RegMasks](Register Reg) -> bool {
    return any_of(RegMasks, [Reg](const uint32_t *RegMask) {
      return MachineOperand::clobbersPhysReg(RegMask, Reg);
    });
  };

  if (!RegMasks.empty()) {
    SmallVector<uint32_t, 32> UsedRegs;
    getUsedRegs(OpenRanges.getVarLocs(), UsedRegs);
    for (uint32_t Reg : UsedRegs) {
      // The VarLocs residing in this register are already in the kill set.
      if (DeadRegs.count(Reg))
        continue;

      // Remove ranges of all clobbered registers. Register masks don't usually
      // list SP as preserved. Assume that call instructions never clobber SP,
      // because some backends (e.g., AArch64) never list SP in the regmask.
      // While the debug info may be off for an instruction or two around
      // callee-cleanup calls, transferring the DEBUG_VALUE across the call is
      // still a better user experience.
      if (Reg == SP)
        continue;
      if (AnyRegMaskKillsReg(Reg)) {
        collectIDsForReg(KillSet, Reg, OpenRanges.getVarLocs());
      }
    }
  }

  // All registers not in the mask may need re-deffing...
  for (unsigned Reg = 1; Reg < TRI->getNumRegs(); ++Reg) {
    if (Reg != SP && AnyRegMaskKillsReg(Reg)) {
      tracker->defReg(Reg, cur_bb, cur_inst);
      if (ttracker)
        ttracker->clobberMloc(Reg, MI.getIterator());
    }
  }

  OpenRanges.erase(KillSet, VarLocIDs);

  if (auto *TPC = getAnalysisIfAvailable<TargetPassConfig>()) {
    auto &TM = TPC->getTM<TargetMachine>();
    if (TM.Options.EnableDebugEntryValues)
      emitEntryValues(MI, OpenRanges, VarLocIDs, Transfers, KillSet);
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
void LiveDebugValues::transferSpillOrRestoreInst(MachineInstr &MI,
                                                 OpenRangesSet &OpenRanges,
                                                 VarLocMap &VarLocIDs,
                                                 TransferMap *Transfers) {
  MachineFunction *MF = MI.getMF();
  TransferKind TKind;
  unsigned Reg;
  Optional<SpillLoc> Loc;

  LLVM_DEBUG(dbgs() << "Examining instruction: "; MI.dump(););

  // First, if there are any DBG_VALUEs pointing at a spill slot that is
  // written to, then close the variable location. The value in memory
  // will have changed.
  VarLocSet KillSet(Alloc);
  if (isSpillInstruction(MI, MF)) {
    Loc = extractSpillBaseRegAndOffset(MI);

    if (ttracker) {
      unsigned mloc = tracker->getSpillMLoc(*Loc);
      if (mloc != 0)
        ttracker->clobberMloc(mloc, MI.getIterator());
    }



    for (uint64_t ID : OpenRanges.getVarLocs()) {
      LocIndex Idx = LocIndex::fromRawInteger(ID);
      const VarLoc &VL = VarLocIDs[Idx];
      if (VL.Kind == VarLoc::SpillLocKind && VL.Loc.SpillLocation == *Loc) {
        // This location is overwritten by the current instruction -- terminate
        // the open range, and insert an explicit DBG_VALUE $noreg.
        //
        // Doing this at a later stage would require re-interpreting all
        // DBG_VALUes and DIExpressions to identify whether they point at
        // memory, and then analysing all memory writes to see if they
        // overwrite that memory, which is expensive.
        //
        // At this stage, we already know which DBG_VALUEs are for spills and
        // where they are located; it's best to fix handle overwrites now.
        KillSet.set(ID);
        VarLoc UndefVL = VarLoc::CreateCopyLoc(VL.MI, LS, 0);
        LocIndex UndefLocID = VarLocIDs.insert(UndefVL);
        if (Transfers)
          Transfers->push_back({&MI, UndefLocID});

      }
    }
    OpenRanges.erase(KillSet, VarLocIDs);
  }

  // Try to recognise spill and restore instructions that may create a new
  // variable location.
  if (isLocationSpill(MI, MF, Reg)) {
    TKind = TransferKind::TransferSpill;
    LLVM_DEBUG(dbgs() << "Recognized as spill: "; MI.dump(););
    LLVM_DEBUG(dbgs() << "Register: " << Reg << " " << printReg(Reg, TRI)
                      << "\n");
  } else {
    if (!(Loc = isRestoreInstruction(MI, MF, Reg)))
      return;
    TKind = TransferKind::TransferRestore;
    LLVM_DEBUG(dbgs() << "Recognized as restore: "; MI.dump(););
    LLVM_DEBUG(dbgs() << "Register: " << Reg << " " << printReg(Reg, TRI)
                      << "\n");
  }

  Loc = extractSpillBaseRegAndOffset(MI);
  if (TKind == TransferKind::TransferSpill) {
    auto id = tracker->readReg(Reg);
    tracker->setSpill(*Loc, id);
    assert(tracker->getSpillMLoc(*Loc) != 0);
    if (ttracker)
      ttracker->transferMlocs(Reg, tracker->getSpillMLoc(*Loc), MI.getIterator());
    tracker->lolwipe(Reg);
  } else {
    auto id = tracker->readSpill(*Loc);
    if (id.LocNo != 0) {
      tracker->setReg(Reg, id);
      assert(tracker->getSpillMLoc(*Loc) != 0);
      if (ttracker)
        ttracker->transferMlocs(tracker->getSpillMLoc(*Loc), Reg, MI.getIterator());
      tracker->lolwipe(*Loc);
    }
  }

  // Check if the register or spill location is the location of a debug value.
  for (uint64_t ID : OpenRanges.getVarLocs()) {
    LocIndex Idx = LocIndex::fromRawInteger(ID);
    const VarLoc &VL = VarLocIDs[Idx];
    if (TKind == TransferKind::TransferSpill && VL.isDescribedByReg() == Reg) {
      LLVM_DEBUG(dbgs() << "Spilling Register " << printReg(Reg, TRI) << '('
                        << VL.Var.getVariable()->getName() << ")\n");
    } else if (TKind == TransferKind::TransferRestore &&
               VL.Kind == VarLoc::SpillLocKind &&
               VL.Loc.SpillLocation == *Loc) {
      LLVM_DEBUG(dbgs() << "Restoring Register " << printReg(Reg, TRI) << '('
                        << VL.Var.getVariable()->getName() << ")\n");
    } else
      continue;
    insertTransferDebugPair(MI, OpenRanges, Transfers, VarLocIDs, Idx, TKind,
                            Reg);
    return;
  }
}

/// If \p MI is a register copy instruction, that copies a previously tracked
/// value from one register to another register that is callee saved, we
/// create new DBG_VALUE instruction  described with copy destination register.
void LiveDebugValues::transferRegisterCopy(MachineInstr &MI,
                                           OpenRangesSet &OpenRanges,
                                           VarLocMap &VarLocIDs,
                                           TransferMap *Transfers) {
  auto DestSrc = TII->isCopyInstr(MI);
  if (!DestSrc)
    return;

  const MachineOperand *DestRegOp = DestSrc->Destination;
  const MachineOperand *SrcRegOp = DestSrc->Source;

  if (!DestRegOp->isDef())
    return;

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
    return;

  // Remember an entry value movement. If we encounter a new debug value of
  // a parameter describing only a moving of the value around, rather then
  // modifying it, we are still able to use the entry value if needed.
  if (isRegOtherThanSPAndFP(*DestRegOp, MI, TRI)) {
    for (uint64_t ID : OpenRanges.getVarLocs()) {
      LocIndex Idx = LocIndex::fromRawInteger(ID);
      const VarLoc &VL = VarLocIDs[Idx];
      if (VL.getEntryValueBackupReg() == SrcReg) {
        LLVM_DEBUG(dbgs() << "Copy of the entry value: "; MI.dump(););
        VarLoc EntryValLocCopyBackup =
            VarLoc::CreateEntryCopyBackupLoc(VL.MI, LS, VL.Expr, DestReg);

        // Stop tracking the original entry value.
        OpenRanges.erase(VL);

        // Start tracking the entry value copy.
        LocIndex EntryValCopyLocID = VarLocIDs.insert(EntryValLocCopyBackup);
        OpenRanges.insert(EntryValCopyLocID, EntryValLocCopyBackup);
        break;
      }
    }
  }

  if (!SrcRegOp->isKill())
    return;

      auto id = tracker->readReg(SrcReg);
      tracker->setReg(DestReg, id);
      if (ttracker)
        ttracker->transferMlocs(SrcReg, DestReg, MI.getIterator());
      tracker->lolwipe(SrcReg);
      return;
}

/// Terminate all open ranges at the end of the current basic block.
bool LiveDebugValues::transferTerminator(MachineBasicBlock *CurMBB,
                                         OpenRangesSet &OpenRanges,
                                         VarLocInMBB &OutLocs,
                                         const VarLocMap &VarLocIDs) {
  bool Changed = false;

  LLVM_DEBUG(for (uint64_t ID
                  : OpenRanges.getVarLocs()) {
    // Copy OpenRanges to OutLocs, if not already present.
    dbgs() << "Add to OutLocs in MBB #" << CurMBB->getNumber() << ":  ";
    VarLocIDs[LocIndex::fromRawInteger(ID)].dump(TRI);
  });
  VarLocSet &VLS = getVarLocsInMBB(CurMBB, OutLocs);
  Changed = VLS != OpenRanges.getVarLocs();
  // New OutLocs set may be different due to spill, restore or register
  // copy instruction processing.
  if (Changed)
    VLS = OpenRanges.getVarLocs();
  OpenRanges.clear();
  return Changed;
}

/// Accumulate a mapping between each DILocalVariable fragment and other
/// fragments of that DILocalVariable which overlap. This reduces work during
/// the data-flow stage from "Find any overlapping fragments" to "Check if the
/// known-to-overlap fragments are present".
/// \param MI A previously unprocessed DEBUG_VALUE instruction to analyze for
///           fragment usage.
/// \param SeenFragments Map from DILocalVariable to all fragments of that
///           Variable which are known to exist.
/// \param OverlappingFragments The overlap map being constructed, from one
///           Var/Fragment pair to a vector of fragments known to overlap.
void LiveDebugValues::accumulateFragmentMap(MachineInstr &MI,
                                            VarToFragments &SeenFragments,
                                            OverlapMap &OverlappingFragments) {
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

    OverlappingFragments.insert({{MIVar.getVariable(), ThisFragment}, {}});
    return;
  }

  // If this particular Variable/Fragment pair already exists in the overlap
  // map, it has already been accounted for.
  auto IsInOLapMap =
      OverlappingFragments.insert({{MIVar.getVariable(), ThisFragment}, {}});
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
          OverlappingFragments.find({MIVar.getVariable(), ASeenFragment});
      assert(ASeenFragmentsOverlaps != OverlappingFragments.end() &&
             "Previously seen var fragment has no vector of overlaps");
      ASeenFragmentsOverlaps->second.push_back(ThisFragment);
    }
  }

  AllSeenFragments.insert(ThisFragment);
}

/// This routine creates OpenRanges.
void LiveDebugValues::process(MachineInstr &MI, OpenRangesSet &OpenRanges,
                              VarLocMap &VarLocIDs, TransferMap *Transfers) {
  transferDebugValue(MI, OpenRanges, VarLocIDs);
  transferRegisterDef(MI, OpenRanges, VarLocIDs, Transfers);
  transferRegisterCopy(MI, OpenRanges, VarLocIDs, Transfers);
  transferSpillOrRestoreInst(MI, OpenRanges, VarLocIDs, Transfers);
}

/// This routine joins the analysis results of all incoming edges in @MBB by
/// inserting a new DBG_VALUE instruction at the start of the @MBB - if the same
/// source variable in all the predecessors of @MBB reside in the same location.
bool LiveDebugValues::join(
    MachineBasicBlock &MBB, VarLocInMBB &OutLocs, VarLocInMBB &InLocs,
    const VarLocMap &VarLocIDs,
    SmallPtrSet<const MachineBasicBlock *, 16> &Visited,
    SmallPtrSetImpl<const MachineBasicBlock *> &ArtificialBlocks,
    VarLocInMBB &PendingInLocs, bool mlocs) {
  LLVM_DEBUG(dbgs() << "join MBB: " << MBB.getNumber() << "\n");
  bool Changed = false;

  VarLocSet InLocsT(Alloc); // Temporary incoming locations.

  // For all predecessors of this MBB, find the set of VarLocs that
  // can be joined.
  int NumVisited = 0;
  for (auto p : MBB.predecessors()) {
    // Ignore backedges if we have not visited the predecessor yet. As the
    // predecessor hasn't yet had locations propagated into it, most locations
    // will not yet be valid, so treat them as all being uninitialized and
    // potentially valid. If a location guessed to be correct here is
    // invalidated later, we will remove it when we revisit this block.
    if (!Visited.count(p)) {
      LLVM_DEBUG(dbgs() << "  ignoring unvisited pred MBB: " << p->getNumber()
                        << "\n");
      continue;
    }
    auto OL = OutLocs.find(p);
    // Join is null in case of empty OutLocs from any of the pred.
    if (OL == OutLocs.end())
      return false;

    // Just copy over the Out locs to incoming locs for the first visited
    // predecessor, and for all other predecessors join the Out locs.
    if (!NumVisited)
      InLocsT = OL->second;
    else
      InLocsT &= OL->second;

    LLVM_DEBUG({
      if (!InLocsT.empty() && !mlocs) {
        for (uint64_t ID : InLocsT)
          dbgs() << "  gathered candidate incoming var: "
                 << VarLocIDs[LocIndex::fromRawInteger(ID)]
                        .Var.getVariable()
                        ->getName()
                 << "\n";
      }
    });

    NumVisited++;
  }

  // Filter out DBG_VALUES that are out of scope.
  VarLocSet KillSet(Alloc);
  bool IsArtificial = ArtificialBlocks.count(&MBB);
  if (!IsArtificial && !mlocs) {
    for (uint64_t ID : InLocsT) {
      LocIndex Idx = LocIndex::fromRawInteger(ID);
      if (!VarLocIDs[Idx].dominates(MBB)) {
        KillSet.set(ID);
        LLVM_DEBUG({
          auto Name = VarLocIDs[Idx].Var.getVariable()->getName();
          dbgs() << "  killing " << Name << ", it doesn't dominate MBB\n";
        });
      }
    }
  }
  InLocsT.intersectWithComplement(KillSet);

  // As we are processing blocks in reverse post-order we
  // should have processed at least one predecessor, unless it
  // is the entry block which has no predecessor.
  assert((NumVisited || MBB.pred_empty()) &&
         "Should have processed at least one predecessor");

  VarLocSet &ILS = getVarLocsInMBB(&MBB, InLocs);
  VarLocSet &Pending = getVarLocsInMBB(&MBB, PendingInLocs);

  // New locations will have DBG_VALUE insts inserted at the start of the
  // block, after location propagation has finished. Record the insertions
  // that we need to perform in the Pending set.
  VarLocSet Diff = InLocsT;
  Diff.intersectWithComplement(ILS);
  Pending.set(Diff);
  ILS.set(Diff);
  NumInserted += Diff.count();
  Changed |= !Diff.empty();

  // We may have lost locations by learning about a predecessor that either
  // loses or moves a variable. Find any locations in ILS that are not in the
  // new in-locations, and delete those.
  VarLocSet Removed = ILS;
  Removed.intersectWithComplement(InLocsT);
  Pending.intersectWithComplement(Removed);
  ILS.intersectWithComplement(Removed);
  NumRemoved += Removed.count();
  Changed |= !Removed.empty();

  return Changed;
}

bool LiveDebugValues::vloc_join(
  const MachineBasicBlock &MBB, VarLocInMBB &VLOCOutLocs,
   VarLocInMBB &VLOCInLocs, lolnumberingt &lolnumbering,
   SmallPtrSet<const MachineBasicBlock *, 16> &VLOCVisited,
   SmallPtrSetImpl<const MachineBasicBlock *> &ArtificialBlocks,
   VarLocInMBB &VLOCPendingInLocs, unsigned cur_bb) {
  LLVM_DEBUG(dbgs() << "join MBB: " << MBB.getNumber() << "\n");
  bool Changed = false;

  VarLocSet InLocsT(Alloc); // Temporary incoming locations.
  VarLocSet toBecomePHIs(Alloc);

  // For all predecessors of this MBB, find the set of VarLocs that
  // can be joined.
  int NumVisited = 0;
  for (auto p : MBB.predecessors()) {
    // Ignore backedges if we have not visited the predecessor yet. As the
    // predecessor hasn't yet had locations propagated into it, most locations
    // will not yet be valid, so treat them as all being uninitialized and
    // potentially valid. If a location guessed to be correct here is
    // invalidated later, we will remove it when we revisit this block.
    if (!VLOCVisited.count(p)) {
      LLVM_DEBUG(dbgs() << "  ignoring unvisited pred MBB: " << p->getNumber()
                        << "\n");
      continue;
    }
    auto OL = VLOCOutLocs.find(p);
    // Join is null in case of empty OutLocs from any of the pred.
    if (OL == VLOCOutLocs.end())
      return false;

    // Just copy over the Out locs to incoming locs for the first visited
    // predecessor, and for all other predecessors join the Out locs.
    if (!NumVisited) {
      InLocsT = OL->second;
      toBecomePHIs = OL->second;
      dbgs() << "vloc_join setting\n";
      InLocsT.dump();
      dbgs() << "\n";
    } else {
      InLocsT &= OL->second;
      toBecomePHIs |= OL->second;
      dbgs() << "vloc_join anding\n";
      OL->second.dump();
      dbgs() << "\n";
    }

    dbgs() << "to get \n";
    InLocsT.dump();

    // xXX jmorse deleted debug statement

    NumVisited++;
  }

  // Erm. We need to produce PHI nodes for vlocs that aren't in the same
  // location. Pick out variables that aren't in InLocsT.
  toBecomePHIs.intersectWithComplement(InLocsT);
  // set for nondeterminism
  MapVector<DebugVariable, unsigned> tophi;
  for (auto ID : toBecomePHIs) {
    tophi.insert(std::make_pair(lolnumbering[ID].first, 0));
  }

  for (auto Var : tophi) {
    InLocsT.set(lolnumbering.insert({Var.first, {{0, 0, 0}, None, {nullptr, false}, cur_bb, ValueRec::PHI}}));
  }
  // Filter out DBG_VALUES that are out of scope.
  VarLocSet KillSet(Alloc);
  bool IsArtificial = ArtificialBlocks.count(&MBB);
  if (!IsArtificial) {
    for (uint64_t ID : InLocsT) {
      auto &lolpair = lolnumbering[ID];
      DebugLoc dl = DebugLoc::get(0, 0, lolpair.first.getVariable()->getScope(), lolpair.first.getInlinedAt());
      // XXX performance fail
      UserValueScopes UVS(dl, LS);
      if (!UVS.dominates(const_cast<MachineBasicBlock *>(&MBB))) {
        KillSet.set(ID);
        // XXX deleted debug statement
      }
    }
  }
  InLocsT.intersectWithComplement(KillSet);

  // As we are processing blocks in reverse post-order we
  // should have processed at least one predecessor, unless it
  // is the entry block which has no predecessor.
  assert((NumVisited || MBB.pred_empty()) &&
         "Should have processed at least one predecessor");

  VarLocSet &ILS = getVarLocsInMBB(&MBB, VLOCInLocs);
  VarLocSet &Pending = getVarLocsInMBB(&MBB, VLOCPendingInLocs);

  // New locations will have DBG_VALUE insts inserted at the start of the
  // block, after location propagation has finished. Record the insertions
  // that we need to perform in the Pending set.
  VarLocSet Diff = InLocsT;
  Diff.intersectWithComplement(ILS);
  Pending.set(Diff);
  ILS.set(Diff);
  NumInserted += Diff.count();
  Changed |= !Diff.empty();

  // We may have lost locations by learning about a predecessor that either
  // loses or moves a variable. Find any locations in ILS that are not in the
  // new in-locations, and delete those.
  VarLocSet Removed = ILS;
  Removed.intersectWithComplement(InLocsT);
  Pending.intersectWithComplement(Removed);
  ILS.intersectWithComplement(Removed);
  NumRemoved += Removed.count();
  Changed |= !Removed.empty();

  return Changed;
}

bool LiveDebugValues::vloc_transfer(VarLocSet &ilocs, VarLocSet &transfer, VarLocSet &olocs, lolnumberingt &lolnumbering) {
  // Eeeerrmmmm...
  // quick implementation then, anything in transfer overrides ilocs. Filter
  // out anything that's been deleted in the meantime.

  VarLocSet new_olocs(Alloc);
  DenseMap<DebugVariable, ValueRec> set;
  for (auto ID : ilocs) {
    set.insert(lolnumbering[ID]);
  }

  for (auto ID : transfer) {
    set.erase(lolnumbering[ID].first);
    set.insert(lolnumbering[ID]);
  }

  // XXX erm, unset any empty locations.
  // XXX XXX are there any now that everything starts with mloc phis?
  for (auto &P : set) {
    if (P.second.Kind == ValueRec::Def && P.second.ID.LocNo == 0)
      continue;
    unsigned id = lolnumbering.idFor(P);
    assert(id != 0);
    new_olocs.set(id);
  }

  bool Changed = new_olocs != olocs;
  olocs = new_olocs;
  return Changed;
}

void LiveDebugValues::resolveMPHIs(mphiremapt &mphiremap, MachineBasicBlock &MBB, VarLocSet &InLocs, VarLocInMBB &MLOCOutLocs, unsigned cur_bb)
{
  // Take a look at any inlocs here that are PHIs; are they really PHIS?
  tracker->reset();
  tracker->loadFromVarLocSet(InLocs, cur_bb);
  std::vector<ValueIDNum> toexamine;
  for (unsigned Idx = 1; Idx < tracker->getNumLocs(); ++Idx) {
    VarLocPos Pos = tracker->getVarLocPos(Idx);
    if (Pos.ID.BlockNo == cur_bb && Pos.ID.InstNo == 0)
      toexamine.push_back(Pos.ID);
  }

  std::vector<ValueIDNum> seen_values = toexamine;
  // Look over predecessors...
  for (auto &p : MBB.predecessors()) {
    tracker->reset();
    tracker->loadFromVarLocSet(getVarLocsInMBB(p, MLOCOutLocs), p->getNumber());
    for (unsigned Idx = 0; Idx < toexamine.size(); ++Idx) {
      VarLocPos outpos = tracker->getVarLocPos(toexamine[Idx].LocNo);
      if (outpos.ID != seen_values[Idx] && outpos.ID != toexamine[Idx] &&
          seen_values[Idx] != toexamine[Idx])
        seen_values[Idx].LocNo = 0;
      else if (outpos.ID != toexamine[Idx])
        seen_values[Idx] = outpos.ID;
    }
  }

  // Any seen values that aren't nulled out means that the only incoming
  // values were the mphi value or one other value. We can remap to that other
  // value.
  for (unsigned Idx = 0; Idx < toexamine.size(); ++Idx) {
    if (seen_values[Idx].LocNo == 0)
      continue;
    //mphiremap.insert(std::make_pair(toexamine[Idx], seen_values[Idx]));
    mphiremap.insert(std::make_pair(std::make_pair(&MBB, seen_values[Idx]), toexamine[Idx]));
  }
}

void LiveDebugValues::resolveVPHIs(vphitomphit &vphitomphi, const mphiremapt &mphiremap, lolnumberingt &lolnumbering, MachineBasicBlock &MBB, VarLocSet &InLocs, VarLocInMBB &VLOCOutLocs, VarLocInMBB &MLOCOutLocs, unsigned cur_bb) {
  // Take a look at each PHI in the inlocs.
  std::vector<std::pair<unsigned, unsigned>> toreplace;
  for (unsigned ID : InLocs) {
    auto &Pair = lolnumbering[ID];
#if 0
    if (Pair.second.Kind == ValueRec::Def) {
      auto it = mphiremap.find(Pair.second.ID);
      if (it != mphiremap.end()) {
        ValueRec tmp = Pair.second;
        tmp.ID = it->second;
        unsigned newID = lolnumbering.insert(std::make_pair(Pair.first, tmp));
        toreplace.push_back(std::make_pair(ID, newID));
      }
    }
#endif

    if (Pair.second.Kind != ValueRec::PHI)
      continue;

    if (Pair.second.BlockPHI != cur_bb) {
      auto it = vphitomphi.find(ID);
      if (it == vphitomphi.end())
        continue;
      toreplace.push_back(std::make_pair(ID, it->second));
      continue;
    }

    bool valid = true;
    unsigned overal_mloc = 0;
    MetaVal meta;
    for (auto p : MBB.predecessors()) {
      const VarLocSet &mlocs = getVarLocsInMBB(p, MLOCOutLocs);
      const VarLocSet &vlocs = getVarLocsInMBB(p, VLOCOutLocs);

      // Find our value num,
      ValueIDNum n_to_find;
      bool found = false;
      for (unsigned ID : vlocs) {
        auto &loc = lolnumbering[ID];
        if (!(loc.first == Pair.first) || loc.second.Kind != ValueRec::Def)
          continue;

        if (overal_mloc != 0 && meta != loc.second.meta) {
          found = false;
          break;
        }

        meta = loc.second.meta;

        // Any PHIs should have been previously resolved; or represent the
        // fact that things are unresolvable.
        n_to_find = loc.second.ID;
        found = true;
        break;
      }
      if (!found) {
        valid = false;
        break;
      }

      found = false;
      unsigned the_mloc = 0;
      for (auto it = mlocs.begin(); it != mlocs.end(); ++it) {
        uint64_t mlocid = *it;
        VarLocPos n = VarLocPos::fromU64(mlocid);
        if (!(n.ID == n_to_find))
          continue;
        the_mloc = n.CurrentLoc;
        found = true;
        break;
      }

      if (!found) {
        valid = false;
        break;
      }
 
      if (overal_mloc != 0 && overal_mloc != the_mloc) {
        valid = false;
        break;
      }
      overal_mloc = the_mloc;
    }

    if (valid && overal_mloc != 0) {
      // Good news, everyone agrees on that mloc. Replace the PHI with an
      // mloc PHI at that position.
      
      ValueIDNum newid = {cur_bb, 0, overal_mloc};
      ValueRec r = {newid, None, meta, 0, ValueRec::Def};
      unsigned newnum = lolnumbering.insert({Pair.first, r});
      // Record pair to mangle later.
      toreplace.push_back(std::make_pair(ID, newnum));
      assert(vphitomphi.find(ID) == vphitomphi.end());
      vphitomphi[ID] = newnum;
    }
  }

  for (auto &P : toreplace) {
    InLocs.reset(P.first);
    InLocs.set(P.second);
  }
}

bool LiveDebugValues::isEntryValueCandidate(
    const MachineInstr &MI, const DefinedRegsSet &DefinedRegs) const {
  assert(MI.isDebugValue() && "This must be DBG_VALUE.");

  // TODO: Add support for local variables that are expressed in terms of
  // parameters entry values.
  // TODO: Add support for modified arguments that can be expressed
  // by using its entry value.
  auto *DIVar = MI.getDebugVariable();
  if (!DIVar->isParameter())
    return false;

  // Do not consider parameters that belong to an inlined function.
  if (MI.getDebugLoc()->getInlinedAt())
    return false;

  // Do not consider indirect debug values (TODO: explain why).
  if (MI.isIndirectDebugValue())
    return false;

  // Only consider parameters that are described using registers. Parameters
  // that are passed on the stack are not yet supported, so ignore debug
  // values that are described by the frame or stack pointer.
  if (!isRegOtherThanSPAndFP(MI.getOperand(0), MI, TRI))
    return false;

  // If a parameter's value has been propagated from the caller, then the
  // parameter's DBG_VALUE may be described using a register defined by some
  // instruction in the entry block, in which case we shouldn't create an
  // entry value.
  if (DefinedRegs.count(MI.getOperand(0).getReg()))
    return false;

  // TODO: Add support for parameters that have a pre-existing debug expressions
  // (e.g. fragments, or indirect parameters using DW_OP_deref).
  if (MI.getDebugExpression()->getNumElements() > 0)
    return false;

  return true;
}

/// Collect all register defines (including aliases) for the given instruction.
static void collectRegDefs(const MachineInstr &MI, DefinedRegsSet &Regs,
                           const TargetRegisterInfo *TRI) {
  for (const MachineOperand &MO : MI.operands())
    if (MO.isReg() && MO.isDef() && MO.getReg())
      for (MCRegAliasIterator AI(MO.getReg(), TRI, true); AI.isValid(); ++AI)
        Regs.insert(*AI);
}

/// This routine records the entry values of function parameters. The values
/// could be used as backup values. If we loose the track of some unmodified
/// parameters, the backup values will be used as a primary locations.
void LiveDebugValues::recordEntryValue(const MachineInstr &MI,
                                       const DefinedRegsSet &DefinedRegs,
                                       OpenRangesSet &OpenRanges,
                                       VarLocMap &VarLocIDs) {
  if (auto *TPC = getAnalysisIfAvailable<TargetPassConfig>()) {
    auto &TM = TPC->getTM<TargetMachine>();
    if (!TM.Options.EnableDebugEntryValues)
      return;
  }

  DebugVariable V(MI.getDebugVariable(), MI.getDebugExpression(),
                  MI.getDebugLoc()->getInlinedAt());

  if (!isEntryValueCandidate(MI, DefinedRegs) ||
      OpenRanges.getEntryValueBackup(V))
    return;

  LLVM_DEBUG(dbgs() << "Creating the backup entry location: "; MI.dump(););

  // Create the entry value and use it as a backup location until it is
  // valid. It is valid until a parameter is not changed.
  DIExpression *NewExpr =
      DIExpression::prepend(MI.getDebugExpression(), DIExpression::EntryValue);
  VarLoc EntryValLocAsBackup = VarLoc::CreateEntryBackupLoc(MI, LS, NewExpr);
  LocIndex EntryValLocID = VarLocIDs.insert(EntryValLocAsBackup);
  OpenRanges.insert(EntryValLocID, EntryValLocAsBackup);
}

/// Calculate the liveness information for the given machine function and
/// extend ranges across basic blocks.
bool LiveDebugValues::ExtendRanges(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "\nDebug Range Extension\n");

  bool Changed = false;
  bool OLChanged = false;
  bool MBBJoined = false;

  VarLocMap VarLocIDs;         // Map VarLoc<>unique ID for use in bitvectors.
  OverlapMap OverlapFragments; // Map of overlapping variable fragments.
  OpenRangesSet OpenRanges(Alloc, OverlapFragments);
                              // Ranges that are open until end of bb.
  VarLocInMBB OutLocs;        // Ranges that exist beyond bb.
  VarLocInMBB InLocs;         // Ranges that are incoming after joining.
  TransferMap Transfers;      // DBG_VALUEs associated with transfers (such as
                              // spills, copies and restores).
  VarLocInMBB PendingInLocs;  // Ranges that are incoming after joining, but
                              // that we have deferred creating DBG_VALUE insts
                              // for immediately.

  VarLocInMBB MLOCOutLocs, MLOCInLocs, MLOCPendingInLocs;

  VarToFragments SeenFragments;

  // Blocks which are artificial, i.e. blocks which exclusively contain
  // instructions without locations, or with line 0 locations.
  SmallPtrSet<const MachineBasicBlock *, 16> ArtificialBlocks;

  DenseMap<unsigned int, MachineBasicBlock *> OrderToBB;
  DenseMap<MachineBasicBlock *, unsigned int> BBToOrder;
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Worklist;
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Pending;

  // Set of register defines that are seen when traversing the entry block
  // looking for debug entry value candidates.
  DefinedRegsSet DefinedRegs;

  // Only in the case of entry MBB collect DBG_VALUEs representing
  // function parameters in order to generate debug entry values for them.
  MachineBasicBlock &First_MBB = *(MF.begin());
  for (auto &MI : First_MBB) {
    collectRegDefs(MI, DefinedRegs, TRI);
      if (MI.isDebugValue())
        recordEntryValue(MI, DefinedRegs, OpenRanges, VarLocIDs);
  }

  // Initialize per-block structures and scan for fragment overlaps.
  for (auto &MBB : MF) {
    PendingInLocs.try_emplace(&MBB, Alloc);

    for (auto &MI : MBB) {
      if (MI.isDebugValue())
        accumulateFragmentMap(MI, SeenFragments, OverlapFragments);
    }
  }

  auto hasNonArtificialLocation = [](const MachineInstr &MI) -> bool {
    if (const DebugLoc &DL = MI.getDebugLoc())
      return DL.getLine() != 0;
    return false;
  };
  for (auto &MBB : MF)
    if (none_of(MBB.instrs(), hasNonArtificialLocation))
      ArtificialBlocks.insert(&MBB);

  LLVM_DEBUG(printVarLocInMBB(MF, OutLocs, VarLocIDs,
                              "OutLocs after initialization", dbgs()));

  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  unsigned int RPONumber = 0;
  for (auto RI = RPOT.begin(), RE = RPOT.end(); RI != RE; ++RI) {
    OrderToBB[RPONumber] = *RI;
    BBToOrder[*RI] = RPONumber;
    Worklist.push(RPONumber);
    ++RPONumber;
  }

  // This is a standard "union of predecessor outs" dataflow problem.
  // To solve it, we perform join() and process() using the two worklist method
  // until the ranges converge.
  // Ranges have converged when both worklists are empty.
  SmallPtrSet<const MachineBasicBlock *, 16> Visited, MLOCVisited;
  while (!Worklist.empty() || !Pending.empty()) {
    // We track what is on the pending worklist to avoid inserting the same
    // thing twice.  We could avoid this with a custom priority queue, but this
    // is probably not worth it.
    SmallPtrSet<MachineBasicBlock *, 16> OnPending;
    LLVM_DEBUG(dbgs() << "Processing Worklist\n");
    while (!Worklist.empty()) {
      MachineBasicBlock *MBB = OrderToBB[Worklist.top()];
      cur_bb = MBB->getNumber();
      cur_inst = 1;
      Worklist.pop();
      MBBJoined = join(*MBB, OutLocs, InLocs, VarLocIDs, Visited,
                       ArtificialBlocks, PendingInLocs, false);
      MBBJoined |= Visited.insert(MBB).second;

     // XXX jmorse
     // Also XXX, do we go around these loops too many times?
      MBBJoined |= join(*MBB, MLOCOutLocs, MLOCInLocs, VarLocIDs, MLOCVisited, ArtificialBlocks, MLOCPendingInLocs, true);
      MLOCVisited.insert(MBB);

      if (MBBJoined) {
        MBBJoined = false;
        Changed = true;
        // Now that we have started to extend ranges across BBs we need to
        // examine spill, copy and restore instructions to see whether they
        // operate with registers that correspond to user variables.
        // First load any pending inlocs.
        OpenRanges.insertFromLocSet(getVarLocsInMBB(MBB, PendingInLocs),
                                    VarLocIDs);
        tracker->loadFromVarLocSet(getVarLocsInMBB(MBB, MLOCInLocs), cur_bb);
        for (auto &MI : *MBB) {
          process(MI, OpenRanges, VarLocIDs, nullptr);
          ++cur_inst;
        }
        OLChanged |= transferTerminator(MBB, OpenRanges, OutLocs, VarLocIDs);

        LLVM_DEBUG(printVarLocInMBB(MF, OutLocs, VarLocIDs,
                                    "OutLocs after propagating", dbgs()));
        LLVM_DEBUG(printVarLocInMBB(MF, InLocs, VarLocIDs,
                                    "InLocs after propagating", dbgs()));

        auto tmpset = tracker->makeVarLocSet();
        auto &replaceset = getVarLocsInMBB(MBB, MLOCOutLocs);
        OLChanged |= tmpset != replaceset;
        replaceset = tmpset;
        tracker->reset();

        if (OLChanged) {
          OLChanged = false;
          for (auto s : MBB->successors())
            if (OnPending.insert(s).second) {
              Pending.push(BBToOrder[s]);
            }
        }
      }
    }
    Worklist.swap(Pending);
    // At this point, pending must be empty, since it was just the empty
    // worklist
    assert(Pending.empty() && "Pending should be empty");
  }

  // vlocs and mlocs: go back over each block, this time tracking the vlocs
  // and building a transfer function between each block. 
  // XXX mv for nondeterminism
  MapVector<unsigned, VLocTracker *> vlocs;
  for (unsigned I = 0; I < MF.size(); ++I)
    vlocs[I] = new VLocTracker();

  // Accumulate things into the vloc tracker.
  for (auto RI = RPOT.begin(), RE = RPOT.end(); RI != RE; ++RI) {
    unsigned Idx = BBToOrder[*RI];
    cur_bb = (*RI)->getNumber();
    Worklist.push(Idx);
    auto *MBB = *RI;
    vtracker = vlocs[Idx];
    tracker->loadFromVarLocSet(getVarLocsInMBB(MBB, MLOCInLocs), cur_bb);
    cur_inst = 1;
    OpenRanges.clear();
    for (auto &MI : *MBB) { // XXX I think the empty open ranges does nufink
      process(MI, OpenRanges, VarLocIDs, nullptr);
      ++cur_inst;
    }
    tracker->reset();
  }

  // OK, we have some transfer functions. Number everything; do data flow.
  UniqueVector<std::pair<DebugVariable, ValueRec>> lolnumbering;
  VarLocInMBB VLOCOutLocs, VLOCInLocs, VLOCPendingInLocs, VLOCTransfer;

  for (auto &It : vlocs) {
    const MachineBasicBlock *MBB = OrderToBB[It.first];
    VarLocSet &transfer = getVarLocsInMBB(MBB, VLOCTransfer);
    for (auto &idx : It.second->Vars) {
      const DebugVariable &Var = idx.first;
      const ValueRec &Rec = idx.second;
      unsigned num = lolnumbering.insert(std::make_pair(Var, Rec));
      transfer.set(num);
    }
  }

  SmallPtrSet<const MachineBasicBlock *, 16> VLOCVisited;
  while (!Worklist.empty() || !Pending.empty()) {
    // We track what is on the pending worklist to avoid inserting the same
    // thing twice.  We could avoid this with a custom priority queue, but this
    // is probably not worth it.
    SmallPtrSet<MachineBasicBlock *, 16> OnPending;
    LLVM_DEBUG(dbgs() << "Processing Worklist\n");
    while (!Worklist.empty()) {
      MachineBasicBlock *MBB = OrderToBB[Worklist.top()];
      cur_bb = MBB->getNumber();
      Worklist.pop();

      MBBJoined = vloc_join(*MBB, VLOCOutLocs, VLOCInLocs, lolnumbering,
                       VLOCVisited,
                       ArtificialBlocks, VLOCPendingInLocs, cur_bb);
      MBBJoined |= VLOCVisited.insert(MBB).second;

      if (MBBJoined) {
        MBBJoined = false;
        Changed = true;

        auto &ilocs = getVarLocsInMBB(MBB, VLOCInLocs);
        auto &transfers = getVarLocsInMBB(MBB, VLOCTransfer);
        auto &olocs = getVarLocsInMBB(MBB, VLOCOutLocs);
        OLChanged = vloc_transfer(ilocs, transfers, olocs, lolnumbering);

        if (OLChanged) {
          OLChanged = false;
          for (auto s : MBB->successors())
            if (OnPending.insert(s).second) {
              Pending.push(BBToOrder[s]);
            }
        }
      }
    }
    Worklist.swap(Pending);
    // At this point, pending must be empty, since it was just the empty
    // worklist
    assert(Pending.empty() && "Pending should be empty");
  }

  for (auto &It : VLOCInLocs) {
    for (auto lala : It.second) {
      auto &var = lolnumbering[lala];
      assert(var.second.Kind != ValueRec::Def || var.second.ID.LocNo != 0);
    }
  }

  // mloc argument only needs the posish -> spills map and the like.
  ttracker = new TransferTracker(TII, tracker, MF);

  // Reprocess all instructions a final time and record transfers. The live-in
  // locations should not change as we've reached a fixedpoint.
  vphitomphit vphitomphi;
  mphiremapt mphiremap;
  for (MachineBasicBlock &MBB : MF) {
    unsigned bbnum = MBB.getNumber();
    resolveMPHIs(mphiremap, MBB, getVarLocsInMBB(&MBB, MLOCInLocs), MLOCOutLocs, bbnum);
  }

  for (MachineBasicBlock &MBB : MF) {
    unsigned bbnum = MBB.getNumber();
    resolveVPHIs(vphitomphi, mphiremap, lolnumbering, MBB, getVarLocsInMBB(&MBB, VLOCInLocs), VLOCOutLocs, MLOCOutLocs, bbnum);
    ttracker->loadInlocs(MBB, lolnumbering, mphiremap, getVarLocsInMBB(&MBB, MLOCInLocs), getVarLocsInMBB(&MBB, VLOCInLocs), bbnum);
    tracker->reset();
    tracker->loadFromVarLocSet(getVarLocsInMBB(&MBB, MLOCInLocs), bbnum);
    tracker->lolremap(&MBB, mphiremap);

    OpenRanges.insertFromLocSet(getVarLocsInMBB(&MBB, PendingInLocs), VarLocIDs);
    for (auto &MI : MBB)
      process(MI, OpenRanges, VarLocIDs, &Transfers);
    OpenRanges.clear();
  }

  for (auto &P : ttracker->Transfers) {
    MachineBasicBlock &MBB = *P.MBB;
    for (auto *MI : P.insts) {
      MBB.insert(P.pos, MI);
    }
  }

  LLVM_DEBUG(printVarLocInMBB(MF, OutLocs, VarLocIDs, "Final OutLocs", dbgs()));
  LLVM_DEBUG(printVarLocInMBB(MF, InLocs, VarLocIDs, "Final InLocs", dbgs()));

  LLVM_DEBUG({
    dbgs() << "At the end of all that...\n";
    for (auto RI = RPOT.begin(), RE = RPOT.end(); RI != RE; ++RI) {
      unsigned id = BBToOrder[*RI];
      MachineBasicBlock *MBB = *RI;
      dbgs() << "In bb " << MBB->getName() << " num " << id << "\n";
      tracker->reset();
      tracker->loadFromVarLocSet(getVarLocsInMBB(MBB, MLOCOutLocs), cur_bb);
      tracker->dump(TRI);

      dbgs() << "variable outlocs\n";
      auto &olocs = getVarLocsInMBB(MBB, VLOCOutLocs);
      for (unsigned ID : olocs) {
        dbgs() << "Var: ";
        dbgs() << lolnumbering[ID].first.getVariable()->getName();
        dbgs() << " locno ";
        lolnumbering[ID].second.dump(TRI);
        dbgs() << "\n";
      }

      dbgs() << "FIN BLOCK\n\n";
    }
  });

  return Changed;
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

  tracker = new MLocTracker(Alloc, TRI->getNumRegs());
  vtracker = nullptr;
  ttracker = nullptr;

  bool Changed = ExtendRanges(MF);
  delete tracker;
  vtracker = nullptr;
  ttracker = nullptr;
  return Changed;
}
