//===- LiveDebugValues.cpp - Tracking Debug Value MIs ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// [ Rename to DebugReachingDefs ]
///
/// This pass propagates variable locations between basic blocks, resolving
/// control flow conflicts between them. The problem is much like SSA
/// construction, where each DBG_VALUE instruction assigns the *value* that
/// a variable has, and every instruction where the variable is in scope uses
/// that variable. The resulting map of instruction-to-value is then translated
/// into a register (or spill) location for each variable over each instruction.
///
/// This pass determines which DBG_VALUE dominates which instructions, or if
/// none do, where values must be merged (like PHI nodes). The added
/// complication is that because codegen has already finished, a PHI node may
/// be needed for a variable location to be correct, but no register or spill
/// slot merges the necessary values. In these circumstances, the variable
/// location is dropped.
///
/// What makes this analysis non-trivial is loops: we cannot tell in advance
/// whether a variable location is live throughout a loop, or whether its
/// location is clobbered (or redefined by another DBG_VALUE), without
/// exploring all the way through.
///
/// To make this simpler we perform two kinds of analysis. First, we identify
/// every value defined by every instruction (ignoring those that only move
/// another value), then compute a map of which values are available for each
/// instruciton. This is stronger than a reaching-def analysis, as we create
/// PHI values where other values merge.
///
/// Secondly, for each variable, we effectively re-construct SSA using each
/// DBG_VALUE as a def. The DBG_VALUEs read a value-number computed by the
/// first analysis from the location they refer to. We can then compute the
/// dominance frontiers of where a variable has a value, and create PHI nodes
/// where they merge.
/// This isn't precisely SSA-construction though, because the function shape
/// is pre-defined. If a variable location requires a PHI node, but no
/// PHI for the relevant values is present in the function (as computed by the
/// first analysis), the location must be dropped.
///
/// Once both are complete, we can pass back over all instructions knowing:
///  * What _value_ each variable should contain, either defined by an
///    instruction or where control flow merges
///  * What the location of that value is (if any).
/// Allowing us to create appropriate live-in DBG_VALUEs, and DBG_VALUEs when
/// a value moves location. After this pass runs, all variable locations within
/// a block should be specified by DBG_VALUEs within that block, allowing
/// DbgEntityHistoryCalculator to focus on individual blocks.
///
/// This pass is able to go fast because the size of the first
/// reaching-definition analysis is proportionate to the working-set size of
/// the function, which the compile tries to keep small. (It's also
/// proportionate to the number of blocks). Additionally, we repeatedly perform
/// the second reaching-definition analysis with only the variables and blocks
/// in a single lexical scope, exploiting their locality.
///
/// NOT IMPLEMENTED (yet): overlapping fragments and entry values.
///
/// Determining where PHIs happen is trickier with this approach, and it comes
/// to a head in the major problem for LiveDebugValues: is a value live-through
/// a loop, or not? Your garden-variety dataflow analysis aims to build a set of
/// facts about a function, however this analysis needs to generate new value
/// numbers at joins.
///
/// To do this, consider a lattice of all definition values, from instructions
/// and from PHIs. Each PHI is characterised by the RPO number of the block it
/// occurs in. Each value pair A, B can be ordered by RPO(A) < RPO(B):
/// with non-PHI values at the top, and any PHI value in the last (by RPO order)
/// block at the bottom.
///
/// (Awkwardly: lower-down-the _lattice_ means a greater RPO _number_. Below,
/// "rank" always refers to the former).
///
/// At any join, for each register, we consider:
///  * All incoming values, and
///  * The PREVIOUS live-in value at this join.
/// If all incoming values agree: that's the live-in value. If they do not, the
/// incoming values are ranked according to the partial order, and the NEXT
/// LOWEST rank after the PREVIOUS live-in value is picked (multiple values of
/// the same rank are ignored as conflicting). If there are no candidate values,
/// or if the rank of the live-in would be lower than the rank of the current
/// blocks PHIs, create a new PHI value.
///
/// Intuitively: if it's not immediately obvious what value a join should result
/// in, we iteratively descend from instruction-definitions down through PHI
/// values, getting closer to the current block each time. If the current block
/// is a loop head, this ordering is effectively searching outer levels of
/// loops, to find a value that's live-through the current loop.
///
/// If the is no value that's live-through this loop, a PHI is created for this
/// location instead. We can't use a lower-ranked PHI because by definition it
/// doesn't dominate the current block. We can't create a PHI value any earlier,
/// because we risk creating a PHI value at a location where values do not in
/// fact merge, thus misrepresenting the truth, and not making the true
/// live-through value for variable locations.
///
/// This algorithm applies to both calculating the availability of values in
/// the first analysis, and the location of variables in the second. However
/// for the second we add an extra dimension of pain: creating a variable
/// location PHI is only valid if, for each incoming edge,
///  * There is a value for the variable on the incoming edge, and
///  * All the edges have that value in the same register.
/// Or put another way: we can only create a variable-location PHI if there is
/// a matching machine-location PHI, the inputs to which are all also the
/// variables location in the predecessor block.
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

// Act more like the old LiveDebugValues, by propagating some locations too
// far and ignoring some transfers.
static cl::opt<bool> EmulateOldLDV(
    "word-wrap-like-word97", cl::Hidden,
    cl::desc("Act like old LiveDebugValues did"),
    cl::init(true));

// Rely on isStoreToStackSlotPostFE and similar to observe all stack spills.
static cl::opt<bool> ObserveAllStackops(
    "observe-all-stack-ops", cl::Hidden,
    cl::desc("Allow non-kill spill and restores"),
    cl::init(false));

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

// This is purely a number that's slightly more strongly typed, to avoid
// passing around raw integers. Identifies a register or spill slot,
// numerically.
enum LocIdx { limin = 0, limax = UINT_MAX };

/// Unique identifier for a value defined by an instruction, as a value type.
/// Casts back and forth to a uint64_t. Probably replacable with something less 
/// bit-constrained. Each value identifies the instruction and machine-location
/// where the value is defined, although there may be no corresponding machine
/// operand for it (ex: regmasks clobbering values). The instructions are 
/// one-based, and definitions that are PHIs have instruction number zero.
class ValueIDNum {
public:
  uint64_t BlockNo : 16;  /// The block where the def happens.
  uint64_t InstNo : 20;   /// The Instruction where the def happens.
                          /// One based, is distance from start of block.
  LocIdx LocNo : 14;      /// The machine-location where the def happens.
 // (No idea why this can work as a LocIdx, it probably shouldn't)

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

// Boilerplate densemapinfo for ValueIDNum.
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

// Boilerplate for our stronger-integer type.
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

/// Meta qualifiers for a value. Pair of whatever expression is used to qualify
/// the the value, and Boolean of whether or not it's indirect.
typedef std::pair<const DIExpression *, bool> MetaVal;

/// Machine location of values tracker class. Listens to the Things being Done
/// by various instructions, and maintains a table of what machine locations
/// have what values (as defined by a ValueIDNum).
/// There are potentially a much larger number of machine locations on the
/// target machine than the actual working-set size of the function. On x86 for
/// example, we're extremely unlikely to want to track values through control
/// or debug registers. To avoid doing so, MLocTracker has several layers of
/// indirection going on, with two kinds of ``location'':
///  * A LocID uniquely identifies a register or spill location, with a
///    predictable value.
///  * A LocIdx is a key (in the database sense) for a LocID and a ValueIDNum.
/// Whenever a location is def'd or used by a MachineInstr, we automagically
/// create a new LocIdx for a location, but not otherwise. This ensures we only
/// account for locations that are actually used or defined. The cost is another
/// vector lookup (of LocID -> LocIdx) over any other implementation. This is
/// fairly cheap, and the compiler tries to reduce the working-set at any one
/// time in the function anyway.
///
/// Register mask operands completely blow this out of the water; I've just
/// piled hacks on top of hacks to get around that.
///
/// A zero LocIdx is reserved for "no value" or "no location".
class MLocTracker {
public:
  MachineFunction &MF;
  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;
  const TargetLowering &TLI;

  /// "Map" of LocIdxes to the ValueIDNums that they store. This is tightly
  /// packed, entries only exist for locations that are being tracked.
  std::vector<ValueIDNum> LocIdxToIDNum;

  /// "Map" of machine location IDs (i.e., raw register or spill number) to the
  /// LocIdx key / number for that location. There are always at least as many
  /// as the number of registers on the target -- if the value in the register
  /// is not being tracked, then the LocIdx value will be zero. New entries are
  /// appended if a new spill slot begins being tracked.
  /// This, and the corresponding reverse map persist for the analysis of the
  /// whole function, and is necessarying for decoding various vectors of
  /// values.
  std::vector<LocIdx> LocIDToLocIdx;

  /// Inverse map of LocIDToLocIdx.
  DenseMap<LocIdx, unsigned> LocIdxToLocID;

  /// Unique-ification of spill locations. Used to number them -- their LocID
  /// number is the index in SpillLocs minus one plus NumRegs.
  UniqueVector<SpillLoc> SpillLocs;

  // Can't remember, something to do with implicitly reading PHIs on the fly. 
  unsigned lolwat_cur_bb;

  /// Cached local copy of the number of registers the target has.
  unsigned NumRegs;

  /// Collection of register mask operands that have been observed. Second part
  /// of pair indicates the instruction that they happened in. Used to
  /// reconstruct where defs happened if we start tracking a location later
  /// on.
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

  /// Produce location ID number for indexing LocIDToLocIdx. Takes the register
  /// or spill number, and flag for whether it's a spill or not.
  unsigned getLocID(unsigned RegOrSpill, bool isSpill) {
    return (isSpill) ? RegOrSpill + NumRegs - 1 : RegOrSpill;
  }

  /// Accessor for reading the value at Idx.
  ValueIDNum getNumAtPos(LocIdx Idx) const {
    assert(Idx < LocIdxToIDNum.size());
    return LocIdxToIDNum[Idx];
  }

  unsigned getNumLocs(void) const {
    return LocIdxToIDNum.size();
  }

  /// Reset all locations to contain a PHI value at the designated block. Used
  /// sometimes for actual PHI values, othertimes to indicate the block entry
  /// value (before any more information is known).
  void setMPhis(unsigned cur_bb) {
    lolwat_cur_bb = cur_bb;
    for (unsigned ID = 1; ID < LocIdxToIDNum.size(); ++ID) {
      LocIdxToIDNum[LocIdx(ID)] = {cur_bb, 0, LocIdx(ID)};
    }
  }

  /// Load values for each location from array of ValueIDNums. Take current
  /// bbnum just in case we read a value from a hitherto untouched register.
  void loadFromArray(uint64_t *Locs, unsigned cur_bb) {
    lolwat_cur_bb = cur_bb;
    // Quickly reset everything to being itself at inst 0, representing a phi.
    for (unsigned ID = 1; ID < LocIdxToIDNum.size(); ++ID) {
      LocIdxToIDNum[LocIdx(ID)] = ValueIDNum::fromU64(Locs[ID]);
    }
  }

  /// Wipe records of what location have what values.
  void reset(void) {
    memset(&LocIdxToIDNum[0], 0, LocIdxToIDNum.size() * sizeof(ValueIDNum));
    Masks.clear();
  }

  /// Clear all data. Destroys the LocID <=> LocIdx map, which makes everything
  /// else in LiveDebugValues uninterpretable.
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

  /// Set a locaiton to a certain value.
  void setMLoc(LocIdx L, ValueIDNum Num) {
    assert(L < LocIdxToIDNum.size());
    LocIdxToIDNum[L] = Num;
  }

  /// Lookup a potentially untracked register ID, storing its LocIdx into Ref.
  /// If ID was not tracked, initialize it to either an mphi value representing
  /// a live-in, or a recent register mask clobber.
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

  /// Record a definition of the specified register at the given block / inst.
  /// This doesn't take a ValueIDNum, because the definition and it's location
  /// are synonymous.
  void defReg(Register r, unsigned bb, unsigned inst) {
    unsigned ID = getLocID(r, false);
    LocIdx &Idx = LocIDToLocIdx[ID];
    bumpRegister(ID, Idx);
    ValueIDNum id = {bb, inst, Idx};
    LocIdxToIDNum[Idx] = id;
  }

  /// Set a register to a value number. To be used if the value number is
  /// known in advance.
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

  /// Reset a register value to zero / empty. Needed to replicate old
  /// LiveDebugValues where a copy to/from a register effectively clears the
  /// contents of the source register. (Values can only have one location in
  /// old LiveDebugValues).
  void WipeRegister(Register r) {
    unsigned ID = getLocID(r, false);
    LocIdx Idx = LocIDToLocIdx[ID];
    LocIdxToIDNum[Idx] = {0, 0, LocIdx(0)};
  }

  /// Determine the LocIdx of an existing register.
  LocIdx getRegMLoc(Register r) {
    unsigned ID = getLocID(r, false);
    return LocIDToLocIdx[ID];
  }

  /// Record a RegMask operand being executed. Defs any register we currently
  /// track, stores a pointer to the mask in case we have to account for it
  /// later.
  void writeRegMask(const MachineOperand *MO, unsigned cur_bb, unsigned inst_id) {
    // Def anything we already have that isn't preserved.
    unsigned SP = TLI.getStackPointerRegisterToSaveRestore();
    // Ensure SP exists, so that we don't override it later.
    unsigned ID = getLocID(SP, false);
    LocIdx &Idx = LocIDToLocIdx[ID];
    bumpRegister(ID, Idx);

    for (auto &P : LocIdxToLocID) {
      // Don't clobber SP, even if the mask says it's clobbered.
      if (P.second != 0 && P.second < NumRegs && P.second != SP &&
          MO->clobbersPhysReg(P.second))
        defReg(P.second, cur_bb, inst_id);
    }
    Masks.push_back(std::make_pair(MO, inst_id));
  }

  /// Set the value stored in a spill location.
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

  /// Read whatever value is in a spill location, or zero if it isn't tracked.
  ValueIDNum readSpill(SpillLoc l) {
    unsigned pos = SpillLocs.idFor(l);
    if (pos == 0)
      // Returning no location -> 0 means $noreg and some hand wavey position
      return {0, 0, LocIdx(0)};

    unsigned L = getLocID(pos, true);
    unsigned LocIdx = LocIDToLocIdx[L];
    return LocIdxToIDNum[LocIdx];
  }

  /// Determine the LocIdx of a spill location.
  LocIdx getSpillMLoc(SpillLoc l) {
    unsigned SpillID = SpillLocs.idFor(l);
    if (SpillID == 0)
      return LocIdx(0);
    unsigned L = getLocID(SpillID, true);
    return LocIDToLocIdx[L];
  }

  /// Return true if Idx is a spill machine location.
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

  /// Create a DBG_VALUE at machine-location MLoc. Qualify it with the
  /// information in meta, for variable Var. Don't insert it anywhere, just
  /// return the builder for it.
  MachineInstrBuilder 
  emitLoc(LocIdx MLoc, const DebugVariable &Var, const MetaVal &meta) {
    DebugLoc DL = DebugLoc::get(0, 0, Var.getVariable()->getScope(), Var.getInlinedAt());
    auto MIB = BuildMI(MF, DL, TII.get(TargetOpcode::DBG_VALUE));

    const DIExpression *Expr = meta.first;
    unsigned Loc = LocIdxToLocID[MLoc];
    if (Loc >= NumRegs) {
      const SpillLoc &Spill = SpillLocs[Loc - NumRegs + 1];
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

/// Class recording the (high level) _value_ of a variable. Identifies either
/// the value of the variable as a ValueIDNum, or a constant MachineOperand.
/// This class also stores meta-information about how the value is qualified.
/// Used to reason about variable values when performing the second 
/// (DebugVariable specific) dataflow analysis.
class ValueRec {
public:
  /// If Kind is Def, the value number that this value is based on.
  ValueIDNum ID;
  /// If Kind is Const, the MachineOperand defining this value.
  Optional<MachineOperand> MO;
  /// Qualifiers for the ValueIDNum above.
  MetaVal meta;

  typedef enum { Def, Const } KindT;
  /// Discriminator for whether this is a constant or an in-program value.
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

/// Types for recording sets of variable fragments that overlap. For a given
/// local variable, we record all other fragments of that variable that could
/// overlap it, to reduce search time.
using FragmentOfVar =
    std::pair<const DILocalVariable *, DIExpression::FragmentInfo>;
using OverlapMap =
    DenseMap<FragmentOfVar, SmallVector<DIExpression::FragmentInfo, 1>>;

/// Collection of DBG_VALUEs observed when traversing a block. Records each
/// variable and the value the DBG_VALUE refers to. Requires the first (machine
/// location) dataflow algorithm to have run already, so that values can be
/// identified.
class VLocTracker {
public:
  /// Map DebugVariable to the latest Value it's defined to have.
  /// Needs to be a mapvector because we determine order-in-the-input-MIR from
  /// the order in this thing.
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

/// Tracker for converting machine values and variable locations into the
/// output of LiveDebugValues: the DBG_VALUEs specifying block live-in
/// locations and transfers within blocks.
/// Operating on a per-block basis, this class takes a (pre-loaded) machine-loc
/// tracker, and must be initialized with the set of variables (and their
/// values) that are live-in to the block. The caller then repeatedly calls
/// process(). TransferTracker picks out machine locations for the live-in
/// variable values (if there is a location) and creates the corresponding
/// DBG_VALUEs. Then, as the block is stepped through, transfers of values
/// between locations are identified and if profitable, a DBG_VALUE created.
///
/// This is where debug use-before-defs would be resolved: a variable with an
/// unavailable value could materialize in the middle of a block, when the
/// value becomes available. Or, we could detect clobbers and re-specify the
/// variable in a backup location. (XXX these are unimplemented).
// 
class TransferTracker {
public:
  const TargetInstrInfo *TII;
  /// This machine-loc tracker is assumed to always contain the up-to-date
  /// value mapping for all machine locations. TransferTracker only reads
  /// information from it. (XXX make it const?)
  MLocTracker *mtracker;
  MachineFunction &MF;

  /// Record of all changed variable locations at a point. Awkwardly, we allow
  /// inserting either before or after the point: MBB != nullptr indicates
  /// it's before, otherwise after.
  struct Transfer {
    MachineBasicBlock::iterator pos;   /// Position to insert DBG_VALUes
    MachineBasicBlock *MBB;            /// non-null if we should insert after.
    std::vector<MachineInstr *> insts; /// Vector of DBG_VALUEs to insert.
  };

  typedef std::pair<LocIdx, MetaVal> LocAndMeta;
  /// Collection of transfers (DBG_VALUEs) to be inserted.
  std::vector<Transfer> Transfers;
  /// Local cache of what-value-is-in-what-LocIdx. Used to identify differences
  /// between TransferTrackers view of variable locations and MLocTrackers. For
  /// example, MLocTracker observes all clobbers, but TransferTracker lazily
  /// does not.
  std::vector<ValueIDNum> VarLocs;

  /// Map from LocIdxes to which DebugVariables are based that location.
  /// Mantained while stepping through the block. Not accurate if
  /// VarLocs[Idx] != mtracker->LocIdxToIDNum[Idx].
  DenseMap<LocIdx, SmallSet<DebugVariable, 4>> ActiveMLocs;
  /// Map from DebugVariable to it's current location and qualifying meta
  /// information. To be used in conjunction with ActiveMLocs to construct
  /// enough information for the DBG_VALUEs for a particular LocIdx.
  DenseMap<DebugVariable, LocAndMeta> ActiveVLocs;

  /// Temporary cache of DBG_VALUEs to be entered into the Transfers collection.
  std::vector<MachineInstr *> PendingDbgValues;

  const TargetRegisterInfo &TRI;
  const BitVector &CalleeSavedRegs;

  TransferTracker(const TargetInstrInfo *TII, MLocTracker *mtracker, MachineFunction &MF, const TargetRegisterInfo &TRI, const BitVector &CalleeSavedRegs) : TII(TII), mtracker(mtracker), MF(MF), TRI(TRI), CalleeSavedRegs(CalleeSavedRegs) { }

  /// Load object with live-in locations. \p mlocs contains the live-in
  /// values in each machine location, while \p vlocs the live-in variable
  /// values. This method picks variable locations for the live-in variables,
  /// creates DBG_VALUEs and puts them in #Transfers, then prepares the other
  /// object fields to track variable locations as we step through the block.
  void loadInlocs(MachineBasicBlock &MBB, uint64_t *mlocs, SmallVectorImpl<std::pair<DebugVariable, ValueRec>> &vlocs, unsigned cur_bb, unsigned NumLocs) {  
    ActiveMLocs.clear();
    ActiveVLocs.clear();
    VarLocs.clear();
    VarLocs.resize(NumLocs);

    auto isCalleeSaved = [&](LocIdx l) {
      unsigned Reg = mtracker->LocIdxToLocID[l];
      for (MCRegAliasIterator RAI(Reg, &TRI, true); RAI.isValid(); ++RAI)
        if (CalleeSavedRegs.test(*RAI))
          return true;
      return false;
    };

    // Map of the preferred location for each value number.
    DenseMap<ValueIDNum, LocIdx> ValueToLoc;

    // Produce a map of value numbers to the current machine locs they live
    // in. When emulating old LiveDebugValues, there should only be one
    // location; when not, we get to pick.
    for (unsigned Idx = 1; Idx < NumLocs; ++Idx) {
      auto VNum = ValueIDNum::fromU64(mlocs[Idx]);
      VarLocs[Idx] = VNum;
      auto it = ValueToLoc.find(VNum);
      // If there's no location for this value yet; or it's a spill, or not a
      /// preferred non-volatile register, then pick this location.
      if (it == ValueToLoc.end() || mtracker->isSpill(it->second) ||
          !isCalleeSaved(it->second))
        ValueToLoc[VNum] = LocIdx(Idx);
    }

    // Now map variables to their picked LocIdxes.
    for (auto Var : vlocs) {
      if (Var.second.Kind == ValueRec::Const) {
        PendingDbgValues.push_back(emitMOLoc(*Var.second.MO, Var.first, Var.second.meta));
        continue;
      }

      // If the value has no location, we can't make a variable location.
      auto ValuesPreferredLoc = ValueToLoc.find(Var.second.ID);
      if (ValuesPreferredLoc == ValueToLoc.end())
        continue;

      LocIdx m = ValuesPreferredLoc->second;
      ActiveVLocs[Var.first] = std::make_pair(m, Var.second.meta);
      ActiveMLocs[m].insert(Var.first);
      assert(m != 0);
      PendingDbgValues.push_back(mtracker->emitLoc(m, Var.first, Var.second.meta));
    }
    flushDbgValues(MBB.begin(), &MBB);
  }

  /// Helper to move created DBG_VALUEs into Transfers collection.
  void flushDbgValues(MachineBasicBlock::iterator pos,
                      MachineBasicBlock *MBB) {
    if (PendingDbgValues.size() > 0) {
      Transfers.push_back({pos, MBB, PendingDbgValues});
      PendingDbgValues.clear();
    }
  }

  /// Handle a DBG_VALUE within a block. Terminate the variables current
  /// location, and record the value its DBG_VALUE refers to, so that we can
  /// detect location transfers later on.
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

    // Check whether our local copy of values-by-location in #VarLocs is out of
    // date. Wipe old tracking data for the location if it's been clobbered in
    // the meantime.
    if (mtracker->getNumAtPos(MLoc) != VarLocs[MLoc]) {
      for (auto &P : ActiveMLocs[MLoc]) {
        ActiveVLocs.erase(P);
      }
      ActiveMLocs[MLoc].clear();
      VarLocs[MLoc] = mtracker->getNumAtPos(MLoc);
    }

    ActiveMLocs[MLoc].insert(Var);
    if (It == ActiveVLocs.end()) {
      ActiveVLocs.insert(std::make_pair(Var, std::make_pair(MLoc, meta)));
    } else {
      It->second.first = MLoc;
      It->second.second = meta;
    }
  }

  /// Explicitly terminate variable locations based on \p mloc. Creates undef
  /// DBG_VALUEs for any variables that were located there, and clears
  /// #ActiveMLoc / #ActiveVLoc tracking information for that location.
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

  /// Transfer variables based on \p src to be based on \dst. This handles
  /// both register copies as well as spills and restores. Creates DBG_VALUEs
  /// describing the movement.
  void transferMlocs(LocIdx src, LocIdx dst, MachineBasicBlock::iterator pos) {
    // Does src still contain the value num we expect? If not, it's been
    // clobbered in the meantime, and our variable locations are stale.
    if (VarLocs[src] != mtracker->getNumAtPos(src))
      return;

    //assert(ActiveMLocs[dst].size() == 0);
    //^^^ Legitimate scenario on account of un-clobbered slot being assigned to?
    ActiveMLocs[dst] = ActiveMLocs[src];
    VarLocs[dst] = VarLocs[src];

    // For each variable based on src; create a location at dst.
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

    // XXX XXX XXX "pretend to be old LDV" means dropping all tracking data
    // about the old location.
    if (EmulateOldLDV)
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

  /// Machine location transfer function, a mapping of locations to new values.
  typedef DenseMap<LocIdx, ValueIDNum> MLocTransferMap;

  /// Live in/out structure for the function: a per-block map of variables to
  /// their values. XXX, better name?
  typedef DenseMap<const MachineBasicBlock *,
                   DenseMap<DebugVariable, ValueRec> *>
      LiveIdxT;

  typedef std::pair<DebugVariable, ValueRec> VarAndLoc;

  /// Vector (per block) of a collection (inner smallvector) of live-ins.
  /// Used as the result type for the variable location dataflow problem.
  typedef SmallVector<SmallVector<VarAndLoc, 8>, 8> LiveInsT;

  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;
  const TargetFrameLowering *TFI;
  BitVector CalleeSavedRegs;
  LexicalScopes LS;

  /// Object to track machine locations as we step through a block. Could
  /// probably be a field rather than a pointer, as it's always used.
  MLocTracker *tracker;
  /// Number of the current block LiveDebugValues is stepping through.
  unsigned cur_bb;
  /// Number of the current instruction LiveDebugValues is evaluating.
  unsigned cur_inst;
  /// Variable tracker -- listens to DBG_VALUEs occurring as LiveDebugValues
  /// steps through a block. Reads the (pre-solved) values at each location
  /// from the MLocTracker obj.
  VLocTracker *vtracker;
  /// Tracker for transfers, listens to DBG_VALUEs and transfers between
  /// locations during stepping, creates new DBG_VALUEs when values are moved
  /// between locations.
  TransferTracker *ttracker;

  /// Blocks which are artificial, i.e. blocks which exclusively contain
  /// instructions without locations, or with line 0 locations.
  SmallPtrSet<const MachineBasicBlock *, 16> ArtificialBlocks;

  // Mapping of blocks to and from their RPOT order.
  DenseMap<unsigned int, MachineBasicBlock *> OrderToBB;
  DenseMap<MachineBasicBlock *, unsigned int> BBToOrder;

  // Map of overlapping variable fragments.
  OverlapMap OverlapFragments;
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

  /// Observe a single instruction while stepping through a block.
  void process(MachineInstr &MI);
  /// Examines whether \p MI is a DBG_VALUE and notifies trackers. 
  /// \returns true if MI was recognized and processed.
  bool transferDebugValue(const MachineInstr &MI);
  /// Examines whether \p MI is copy instruction, and notifies trackers.
  /// \returns true if MI was recognized and processed.
  bool transferRegisterCopy(MachineInstr &MI);
  /// Examines whether \p MI is stack spill or restore  instruction, and
  /// notifies trackers. \returns true if MI was recognized and processed.
  bool transferSpillOrRestoreInst(MachineInstr &MI);
  /// Examines \p MI for any registers that it defines, and notifies trackers.  
  /// \returns true if MI was recognized and processed.
  void transferRegisterDef(MachineInstr &MI);

  void accumulateFragmentMap(MachineInstr &MI);

  /// Step through the function, recording register definitions and movements
  /// in an MLocTracker. Convert the observations into a per-block transfer
  /// function in \p MLocTransfer, suitable for using with the first (machine
  /// location of values) dataflow problem.
  void produce_mloc_transfer_function(MachineFunction &MF,
                           std::vector<MLocTransferMap> &MLocTransfer,
                           unsigned MaxNumBlocks);

  /// Solve the machine location of values dataflow problem. Takes as input the
  /// transfer functions in \p MLocTransfer and (implicitly) the location maps
  /// in the MLocTracker object. Writes the output live-in and live-out arrays
  /// to the (initialized to zero) multidimensional arrays in \p MInLocs and
  /// \p MOutLocs. The outer dimension is indexed by block number, the inner
  /// by LocIdx. 
  void mloc_dataflow(uint64_t **MInLocs, uint64_t **MOutLocs,
                     std::vector<MLocTransferMap> &MLocTransfer);

  /// Perform a control flow join (lattice value meet) of the values in
  /// machine locations at \p MBB. Follows the algorithm described in the
  /// file-comment, reading live-outs of predecessors from \p OutLocs, the
  /// current live ins from \p InLocs, and assigning the newly computed live ins
  /// back into \p InLocs. \returns true if a change was made.
  /// \p BBNumToRPO maps block numbers (getNumber) to RPO numbers.
  bool mloc_join(MachineBasicBlock &MBB,
                 SmallPtrSet<const MachineBasicBlock *, 16> &Visited,
                 uint64_t **OutLocs, uint64_t *InLocs,
                 DenseMap<unsigned, unsigned> &BBNumToRPO);

  /// Solve the variable value dataflow problem, for a single lexical scope.
  /// Uses the algorithm from the file comment to resolve control flow joins,
  /// although there are extra hacks, see vloc_join_location. Reads the
  /// locations of values from the \p MInLocs and \p MOutLocs arrays (see
  /// mloc_dataflow) and reads the variable values transfer function from
  /// \p AllTheVlocs. Live-in and Live-out variable values are stored locally,
  /// with the live-ins permanently stored to \p Output once the fixedpoint is
  /// reached.
  /// \p VarsWeCareAbout contains a collection of the variables in \p Scope
  /// that we should be tracking.
  /// \p AssignBlocks contains the set of blocks that aren't in \p Scope, but
  /// which do contain DBG_VALUEs, which old LiveDebugValues tracked locations
  /// through.
  void vloc_dataflow(
      const LexicalScope *Scope,
      const SmallSet<DebugVariable, 4> &VarsWeCareAbout,
      SmallPtrSetImpl<MachineBasicBlock *> &AssignBlocks,
      LiveInsT &Output,
      uint64_t **MOutLocs, uint64_t **MInLocs,
      SmallVectorImpl<VLocTracker> &AllTheVLocs);

  /// Compute the live-ins to a block, considering control flow merges according
  /// to the method in the file comment. Live out and live in variable values
  /// are stored in \p VLOCOutLocs and \p VLOCInLocs, while machine value
  /// locations are in \p MOutLocs and \p MInLocs. The live-ins for \p MBB are
  /// computed and stored into \p VLOCInLocs. \returns true if the live-ins
  /// are modified. Delegates most logic for merging to \ref vloc_join_location.
  bool vloc_join(MachineBasicBlock &MBB, LiveIdxT &VLOCOutLocs,
                 LiveIdxT &VLOCInLocs,
                 SmallPtrSet<const MachineBasicBlock *, 16> *VLOCVisited,
                 unsigned cur_bb, const SmallSet<DebugVariable, 4> &AllVars,
                 uint64_t **MInLocs, uint64_t **MOutLocs,
                 SmallPtrSet<const MachineBasicBlock *, 8> &NonAssignBlocks);

  /// Perform location merge for a single variable at a particular block,
  /// between two individual predecessor values. \ref vloc_join picks one
  /// predecessor live-out location as a "base" live-in, then merges all the
  /// other locations into it with this method.
  /// \p InLoc One of the incoming values,
  /// \p OLoc The other incoming value.
  /// \p PrevInLocs Map of what the previous live-in values were.
  /// \p CurVar the DebugVariable we're working with.
  /// \p ThisIsABackEdge True if \p OLoc is a backedge.
  /// \returns true if the locations are reconciled, false if there is an
  /// unresolvable location conflict.
  bool vloc_join_location(MachineBasicBlock &MBB, ValueRec &InLoc,
                          ValueRec &OLoc, uint64_t *InLocOutLocs,
                          uint64_t *OLOutlocs,
                          const LiveIdxT::mapped_type PrevInLocs, // is ptr
                          const DebugVariable &CurVar, bool ThisIsABackEdge);


  /// Given the solutions to the two dataflow problems, machine value locations
  /// in \p MInLocs and live-in variable values in \p SavedLiveIns, runs the
  /// TransferTracker class over the function to produce live-in and transfer
  /// DBG_VALUEs, then inserts them. Groups of DBG_VALUEs are inserted in the
  /// order given by AllVarsNumbering -- this could be any stable order, but
  /// right now "order of appearence in function, when explored in RPO", so
  /// that we can compare explictly against old LiveDebugValues.
  void emit_locations(MachineFunction &MF, LiveInsT SavedLiveIns, uint64_t **MInLocs, DenseMap<DebugVariable, unsigned> &AllVarsNumbering);

  /// Boilerplate computation of some initial sets, artifical blocks and
  /// RPOT block ordering.
  void initial_setup(MachineFunction &MF);

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
  void dump_mloc_transfer(const MLocTransferMap &mloc_transfer) const;

  bool isCalleeSaved(LocIdx l) {
    unsigned Reg = tracker->LocIdxToLocID[l];
    for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
      if (CalleeSavedRegs.test(*RAI))
        return true;
    return false;
  }
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

  // If there are no instructions in this lexical scope, do no location tracking
  // at all, this variable shouldn't have a legitimate location range.
  auto *Scope = LS.findLexicalScope(MI.getDebugLoc().get());
  if (Scope == nullptr)
    return true; // handled it; by doing nothing

  const MachineOperand &MO = MI.getOperand(0);

  // MLocTracker needs to know that this register is read, even if it's only
  // read by a debug inst.
  if (MO.isReg() && MO.getReg() != 0)
    (void)tracker->readReg(MO.getReg());

  // If we're preparing for the second analysis (variables), the values in
  // machine locations are already solved, and we report this DBG_VALUE and the
  // value it refers to to VLocTracker.
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

  // If performing final tracking of transfers, report this variable definition
  // to the TransferTracker too.
  if (ttracker)
    ttracker->redefVar(MI);
  return true;
}

void LiveDebugValues::transferRegisterDef(MachineInstr &MI) {
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

  // Tell MLocTracker about all definitions, of regmasks and otherwise.
  for (uint32_t DeadReg : DeadRegs)
    tracker->defReg(DeadReg, cur_bb, cur_inst);

  for (auto *MO : RegMaskPtrs)
    tracker->writeRegMask(MO, cur_bb, cur_inst);
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

  // XXX FIXME: On x86, isStoreToStackSlotPostFE returns '1' instead of an
  // actual register number.
  if (ObserveAllStackops) {
    int FI;
    Reg = TII->isStoreToStackSlotPostFE(MI, FI);
    return Reg != 0;
  }

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

bool LiveDebugValues::transferSpillOrRestoreInst(MachineInstr &MI) {
  // XXX -- it's too difficult to implement old LiveDebugValues' stack location
  // limitations under the new model. Therefore, when comparing them, compare
  // versions that don't attempt spills or restores at all.
  if (EmulateOldLDV)
    return false;

  MachineFunction *MF = MI.getMF();
  unsigned Reg;
  Optional<SpillLoc> Loc;

  LLVM_DEBUG(dbgs() << "Examining instruction: "; MI.dump(););

  // First, if there are any DBG_VALUEs pointing at a spill slot that is
  // written to, terminate that variable location. The value in memory
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

    // If the location is empty, produce a phi, signify it's the live-in value.
    if (id.LocNo == 0)
      id = {cur_bb, 0, tracker->getRegMLoc(Reg)};

    tracker->setSpill(*Loc, id);
    assert(tracker->getSpillMLoc(*Loc) != 0);

    // Tell TransferTracker about this spill, produce DBG_VALUEs for it.
    if (ttracker)
      ttracker->transferMlocs(tracker->getRegMLoc(Reg), tracker->getSpillMLoc(*Loc), MI.getIterator());

    // Old LiveDebugValues would, at this point, stop tracking the source
    // register of the store.
    if (EmulateOldLDV) {
      for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
        tracker->defReg(*RAI, cur_bb, cur_inst);
    }
  } else {
    if (!(Loc = isRestoreInstruction(MI, MF, Reg)))
      return false;

    // Is there a value to be restored?
    auto id = tracker->readSpill(*Loc);
    if (id.LocNo != 0) {
      // XXX -- can we recover sub-registers of this value? Until we can, first
      // overwrite all defs of the register being restored to.
      for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
        tracker->defReg(*RAI, cur_bb, cur_inst);

      // Now override the reg we're restoring to.
      tracker->setReg(Reg, id);
      assert(tracker->getSpillMLoc(*Loc) != 0);

      // Report this restore to the transfer tracker too.
      if (ttracker)
        ttracker->transferMlocs(tracker->getSpillMLoc(*Loc), tracker->getRegMLoc(Reg), MI.getIterator());
    } else {
      // There isn't anything in the location; not clear if this is a code path
      // that still runs. Def this register anyway just in case.
      for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
        tracker->defReg(*RAI, cur_bb, cur_inst);

      // Set the restored value to be a machine phi number, signifying that it's
      // whatever the spills live-in value is in this block.
      LocIdx l = tracker->getSpillMLoc(*Loc);
      id = {cur_bb, 0, l};
      tracker->setReg(Reg, id);
    }
  }
  return true;
}

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

  // Ignore identity copies. Yep, these make it as far as LiveDebugValues.
  if (SrcReg == DestReg)
    return true;

  // For emulating old LiveDebugValues:
  // We want to recognize instructions where destination register is callee
  // saved register. If register that could be clobbered by the call is
  // included, there would be a great chance that it is going to be clobbered
  // soon. It is more likely that previous register location, which is callee
  // saved, is going to stay unclobbered longer, even if it is killed.
  //
  // For new LiveDebugValues, we can track multiple locations, so ignore this
  // condition.
  if (EmulateOldLDV && !isCalleeSavedReg(DestReg))
    return false;

  // Old LiveDebugValues only followed killing copies.
  if (EmulateOldLDV && !SrcRegOp->isKill())
    return false;

  // We have to follow identity copies, as DbgEntityHistoryCalculator only
  // sees the defs. XXX is this code path still taken?
  auto id = tracker->readReg(SrcReg);
  tracker->setReg(DestReg, id);

  // Only produce a transfer of DBG_VALUE within a block where old LDV
  // would have. We might make use of the additional value tracking in some
  // other way, later.
  if (ttracker && isCalleeSavedReg(DestReg) && SrcRegOp->isKill())
    ttracker->transferMlocs(tracker->getRegMLoc(SrcReg), tracker->getRegMLoc(DestReg), MI.getIterator());

  // Old LiveDebugValues would quit tracking the old location after copying.
  if (EmulateOldLDV && SrcReg != DestReg)
    tracker->WipeRegister(SrcReg);

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

void LiveDebugValues::process(MachineInstr &MI) {
  // Try to interpret an MI as a debug or transfer instruction. Only if it's
  // none of these should we interpret it's register defs as new value
  // definitions.
  if (transferDebugValue(MI))
    return;
  if (transferRegisterCopy(MI))
    return;
  if (transferSpillOrRestoreInst(MI))
    return;
  transferRegisterDef(MI);
}

void LiveDebugValues::produce_mloc_transfer_function(MachineFunction &MF,
                           std::vector<MLocTransferMap> &MLocTransfer,
                           unsigned MaxNumBlocks)
{
  // Because we try to optimize around register mask operands by ignoring regs
  // that aren't currently tracked, we set up something ugly for later: RegMask
  // operands that are seen earlier than the first use of a register, still need
  // to clobber that location in the transfer function. But this information
  // isn't actively recorded. Instead, we track each RegMask used in each block,
  // and accumulated the clobbered but untracked registers in each block into
  // the following bitvector. Later, if new values are tracked, we can add
  // appropriate clobbers.
  std::vector<BitVector> BlockMasks;
  BlockMasks.resize(MaxNumBlocks);

  // Reserve one bit per register for the masks described above.
  unsigned BVWords = MachineOperand::getRegMaskSize(TRI->getNumRegs());
  for (auto &BV : BlockMasks)
    BV.resize(TRI->getNumRegs(), true);

  // Step through all instructions and inhale the transfer function.
  for (auto &MBB : MF) {
    // Object fields that are read by trackers to know where we are in the
    // function.
    cur_bb = MBB.getNumber();
    cur_inst = 1;

    // Set all tracked locations to a PHI value. For transfer function
    // production only, this signifies the live-in value to the block.
    tracker->reset();
    tracker->setMPhis(cur_bb);

    // Step through each instruction in this block.
    for (auto &MI : MBB) {
      process(MI);
      // Also accumulate fragment map.
      if (MI.isDebugValue())
        accumulateFragmentMap(MI);
      ++cur_inst;
    }

    // Produce the transfer function, a map of location to new value. If any
    // location has the live-in phi value from the start of the block, it's
    // live-through and doesn't need recording in the transfer function.
    for (unsigned IdxNum = 1; IdxNum < tracker->getNumLocs(); ++IdxNum) {
      LocIdx Idx = LocIdx(IdxNum);
      ValueIDNum P = tracker->getNumAtPos(Idx);
      if (P.InstNo == 0 && P.LocNo == Idx)
        continue;

      MLocTransfer[cur_bb][Idx] = P;
    }

    // Accumulate any bitmask operands into the clobberred reg mask for this 
    // block.
    for (auto &P : tracker->Masks) {
      BlockMasks[cur_bb].clearBitsNotInMask(P.first->getRegMask(), BVWords);
    }
  }

  // Compute a bitvector of all the registers that are tracked in this block.
  const TargetLowering *TLI = MF.getSubtarget().getTargetLowering();
  unsigned SP = TLI->getStackPointerRegisterToSaveRestore();
  BitVector UsedRegs(TRI->getNumRegs());
  for (auto &P : tracker->LocIdxToLocID) {
    if (P.first == 0 || P.second >= TRI->getNumRegs() || P.second == SP)
      continue;
    UsedRegs.set(P.second);
  }

  // Check that any regmask-clobber of a register that gets tracked, is not
  // live-through in the transfer function. It needs to be clobbered at the
  // very least.
  // XXX, this doesn't account for setting a reg and then clobbering it
  // afterwards, although I guess then the reg would be tracked?
  // XXX, also, no-entry should be turned into a clobber too, right?
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

bool LiveDebugValues::mloc_join(
    MachineBasicBlock &MBB,
    SmallPtrSet<const MachineBasicBlock *, 16> &Visited,
    uint64_t **OutLocs, uint64_t *InLocs,
    DenseMap<unsigned, unsigned> &BBNumToRPO) {
  LLVM_DEBUG(dbgs() << "join MBB: " << MBB.getNumber() << "\n");
  bool Changed = false;

  // Collect predecessors that have been visited. Anything that hasn't been
  // visited yet is a backedge on the first iteration, and the meet of it's
  // lattice value for all locations will be unaffected.
  SmallVector<const MachineBasicBlock *, 8> BlockOrders;
  for (auto p : MBB.predecessors()) {
    if (Visited.count(p)) {
      BlockOrders.push_back(p);
    }
  }

  // Visit predecessors in RPOT order.
  auto Cmp = [&](const MachineBasicBlock *A, const MachineBasicBlock *B) {
   return BBToOrder.find(A)->second < BBToOrder.find(B)->second;
  };
  llvm::sort(BlockOrders.begin(), BlockOrders.end(), Cmp);

  // Skip entry block.
  if (BlockOrders.size() == 0)
    return false;

  // Step through all locations, then look at each predecessor and detect
  // disagreements.
  unsigned this_block_rpot = BBToOrder.find(&MBB)->second;
  for (unsigned Idx = 1; Idx < tracker->getNumLocs(); ++Idx) {
    // Pick out the first predecessors live-out value for this location. It's
    // guaranteed to be not a backedge, as we order by RPO.
    uint64_t base = OutLocs[BlockOrders[0]->getNumber()][Idx];

    // Some flags for whether there's a disagreement, and whether it's a
    // disagreement with a backedge or not.
    bool disagree = false;
    bool non_be_disagree = false;

    for (auto *MBB : BlockOrders) { // XXX tests against itself.
      if (base != OutLocs[MBB->getNumber()][Idx]) {
        // Live-out of a predecessor disagrees with the first predecessor.
        disagree = true;

        // Test whether it's a disagreemnt in the backedges or not.
        if (BBToOrder.find(MBB)->second < this_block_rpot) // might be self b/e
          non_be_disagree = true;
      }
    }

    bool over_ride = false;
    if (disagree && !non_be_disagree && ValueIDNum::fromU64(InLocs[Idx]).LocNo != 0) {
      // Only the backedges disagree, and we previously agreed on some value
      // because we set the Live-In to be nonzero. Consider demoting the livein
      // lattice value, as per the file level comment. The value we consider
      // demoting to is the value that the non-backedge predecessors agree on.
      // The order of values is that non-PHIs are \top, a PHI at this block 
      // \bot, and phis between the two are ordered by their RPO number.
      // If there's no agreement, or we've already demoted to this PHI value
      // before, replace with a PHI value at this block.

      // Calculate order numbers: zero means normal def, nonzero means RPO
      // number.
      ValueIDNum base_id = ValueIDNum::fromU64(base);
      unsigned base_block = BBNumToRPO[base_id.BlockNo] + 1;
      if (base_id.InstNo != 0)
        base_block = 0;

      ValueIDNum inloc_id = ValueIDNum::fromU64(InLocs[Idx]);
      unsigned inloc_block = BBNumToRPO[inloc_id.BlockNo] + 1;
      if (inloc_id.InstNo != 0)
        inloc_block = 0;

      // Should we ignore the disagreeing backedges, and override with the
      // value the other predecessors agree on (in "base")?
      unsigned this_block = BBNumToRPO[MBB.getNumber()] + 1;
      if (base_block > inloc_block && base_block < this_block) {
        // Override.
        over_ride = true;
      }
    }
    // else: if we disagree in the non-backedges, then this is definitely
    // a control flow merge where different values merge. Make it a PHI.

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
    uint64_t **MOutLocs, std::vector<MLocTransferMap> &MLocTransfer) {
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Worklist, Pending;

  DenseMap<unsigned, unsigned> BBNumToRPO;

  for (unsigned int I = 0; I < BBToOrder.size(); ++I) {
    Worklist.push(I);
    BBNumToRPO[OrderToBB[I]->getNumber()] = I;
  }

  tracker->reset();

  // Set inlocs for entry block -- each as a PHI at the entry block. Represents
  // the incoming value to the function.
  tracker->setMPhis(0);
  for (unsigned Idx = 1; Idx < tracker->getNumLocs(); ++Idx) {
    ValueIDNum Val = tracker->getNumAtPos(LocIdx(Idx));
    uint64_t ID = Val.asU64();
    MInLocs[0][Idx] = ID;
  }

  SmallPtrSet<const MachineBasicBlock *, 16> Visited;
  while (!Worklist.empty() || !Pending.empty()) {
    // We track what is on the pending worklist to avoid inserting the same
    // thing twice. We could avoid this with a custom priority queue, but this
    // is probably not worth it.
    SmallPtrSet<MachineBasicBlock *, 16> OnPending;

    // Vector for storing the evaluated block transfer function.
    SmallVector<std::pair<LocIdx, ValueIDNum>, 32> toremap;

    while (!Worklist.empty()) {
      MachineBasicBlock *MBB = OrderToBB[Worklist.top()];
      cur_bb = MBB->getNumber();
      Worklist.pop();

      // Join the values in all predecessor blocks.
      bool InLocsChanged = mloc_join(*MBB, Visited, MOutLocs, MInLocs[cur_bb],
                                     BBNumToRPO);
      InLocsChanged |= Visited.insert(MBB).second;

      // Don't examine transfer function if we've visited this loc at least
      // once, and inlocs haven't changed.
      if (!InLocsChanged)
        continue;

      // Load the current set of live-ins into MLocTracker.
      tracker->loadFromArray(MInLocs[cur_bb], cur_bb);

      // Each element of the transfer function can be a new def, or a read of
      // a live-in value. Evaluate each element, and store to "toremap".
      toremap.clear();
      for (auto &P : MLocTransfer[cur_bb]) {
        ValueIDNum NewID = {0, 0, LocIdx(0)};
        if (P.second.BlockNo == cur_bb && P.second.InstNo == 0) {
          // This is a movement of whatever was live in. Read it.
          NewID = tracker->getNumAtPos(P.second.LocNo);
        } else {
          // It's a def. Just set it.
          assert(P.second.BlockNo == cur_bb || P.second.LocNo == 0);
          NewID = P.second;
        }
        toremap.push_back(std::make_pair(P.first, NewID));
      }

      // Commit the transfer function changes into mloc tracker, which
      // transforms the contents of the MLocTracker into the live-outs.
      for (auto &P : toremap)
        tracker->setMLoc(P.first, P.second);

      // Now copy out-locs from mloc tracker into out-loc vector, checking
      // whether changes have occurred. These changes can have come from both
      // the transfer function, and mloc_join.
      bool OLChanged = false;
      for (unsigned Idx = 1; Idx < tracker->getNumLocs(); ++Idx) {
        uint64_t ID = tracker->getNumAtPos(LocIdx(Idx)).asU64();
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

  // Once all the live-ins don't change on mloc_join(), we've reached a
  // fixedpoint.
}

bool LiveDebugValues::vloc_join_location(MachineBasicBlock &MBB,
                        ValueRec &InLoc,
                        ValueRec &OLoc, uint64_t *InLocOutLocs,
                        uint64_t *OLOutLocs,
                        const LiveIdxT::mapped_type PrevInLocs, // ptr
                        const DebugVariable &CurVar, bool ThisIsABackEdge)
{
  // This method checks whether InLoc and OLoc, the locations of a variable
  // in two predecessor blocks, are reconcilable. The answer can be "yes", "no",
  // and "yes when downgraded to a PHI value".
  // Unfortuantely this is mega-complex, because as well as deciding whether
  // two values can merge or be PHI'd, we also have to verify that if a PHI is
  // to be used, that the corresponding machine-location PHI has the correct
  // inputs from the predecessors.
  // InLoc is never a backedge; the rest of the join problem usually hinges on
  // whether OLoc is a backedge or not.
  // Some decisions are left to TransferTracker: if we can determine a value
  // that isn't an mphi, we can safely leave TransferTracker to pick a location
  // for it.

  unsigned cur_bb = MBB.getNumber();
  bool EarlyBail = false;

  // Lambda to pick a machine location for a value, if we decide to use a PHI
  // and merge them. This could be much more sophisticated, but right now
  // is good enough. When emulating old LiveDebugValues, there should only be
  // one candidate location for a value anyway.
  auto FindLocInLocs = [&](uint64_t *OutLocs, const ValueIDNum &ID) -> LocIdx {
    unsigned NumLocs = tracker->getNumLocs();
    LocIdx theloc = LocIdx(0);
    for (unsigned i = 0; i < NumLocs; ++i) {
      if (OutLocs[i] == ID.asU64()) {
        if (theloc != 0) {
          // Prefer non-spills
          if (tracker->isSpill(theloc))
            theloc = LocIdx(i);
          else if (!isCalleeSaved(theloc))
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

  // Specialisations of that lambda for InLoc and OLoc.
  auto FindInInLocs = [&](const ValueIDNum &ID) -> LocIdx {
    return FindLocInLocs(InLocOutLocs, ID);
  };
  auto FindInOLocs = [&](const ValueIDNum &ID) -> LocIdx {
    return FindLocInLocs(OLOutLocs, ID);
  };

  // Different kinds (const/def)? Definite no.
  EarlyBail |= InLoc.Kind != OLoc.Kind;

  // Trying to join constants is very simple. Plain join on the constant
  // value. Set EarlyBail if they differ.
  EarlyBail |=
     (InLoc.Kind == OLoc.Kind && InLoc.Kind == ValueRec::Const &&
      !InLoc.MO->isIdenticalTo(*OLoc.MO));

  // Meta disagreement -> bail early. We wouldn't be able to produce a
  // DBG_VALUE that reconciled the meta information.
  EarlyBail |= (InLoc.meta != OLoc.meta);

  // LocNo == 0 (undef) -> bail early.
  EarlyBail |=
     (InLoc.Kind == OLoc.Kind && InLoc.Kind == ValueRec::Def &&
      OLoc.ID.LocNo == 0);

  // Bail out if early bail signalled.
  if (EarlyBail) {
    return false;
  } else if (InLoc.Kind == ValueRec::Const) {
    // If both are constants and we didn't early-bail, they're the same.
    return true;
  }

  // This is a join for "values". Two important facts: is this a backedge, and
  // does InLocs refer to a machine-location PHI already?
  assert(InLoc.Kind == ValueRec::Def);
  ValueIDNum &InLocsID = InLoc.ID;
  ValueIDNum &OLID = OLoc.ID;
  bool ThisIsAnMPHI = InLocsID.BlockNo == cur_bb && InLocsID.InstNo == 0;

  // Find a location for the OLID in its out-locs.
  LocIdx OLIdx = FindInOLocs(OLID);

  // Everything is massively different for backedges. Try not-be's first.
  if (!ThisIsABackEdge) {
    // If both values agree, no more work is required, and a location can be
    // picked for the value when DBG_VALUEs are created.
    // However if they disagree, or the value is a PHI in this block, then
    // we may need to create a new PHI, or verify that the correct values flow
    // into the machine-location PHI.
    if (InLoc == OLoc && !ThisIsAnMPHI)
      return true;

    // If we're non-identical and there's no mphi, definitely can't merge.
    // XXX document that InLoc is always the mphi, if ther eis noe.
    if (InLoc != OLoc && !ThisIsAnMPHI)
      return false;

    // Otherwise, we're definitely an mphi, and need to prove that the
    // location from OLoc goes into it. Because we're an mphi, we know
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

  // If the backedge value has no location, definitely can't merge.
  if (OLIdx == 0)
    return false;

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

  // We're not identical, values are merging and we haven't yet picked an mphi
  // starting at this block. Follow the file level comment algorithm, and
  // consider demoting the incoming PHI value instead. Or, downgrade to using
  // an mphi starting in this block.

  auto ILS_It = PrevInLocs->find(CurVar);
  if (ILS_It == PrevInLocs->end() || ILS_It->second.Kind != ValueRec::Def)
    // This is the first time around as there's no in-loc, and if there are
    // disagreements, they won't be due to back edges, thus it's immediately
    // fatal to this location.
    return false;

  // XXX XXX XXX should be RPO order.
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
          else if (!isCalleeSaved(theloc))
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
    LiveInsT &Output,
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
  if (EmulateOldLDV)
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

void LiveDebugValues::dump_mloc_transfer(const MLocTransferMap &mloc_transfer) const {
  for (auto &P : mloc_transfer) {
    std::string foo = tracker->LocIdxToName(P.first);
    std::string bar = tracker->IDAsString(P.second);
    dbgs() << "Loc " << foo << " --> " << bar << "\n";
  }
}

void LiveDebugValues::emit_locations(MachineFunction &MF, LiveInsT SavedLiveIns, uint64_t **MInLocs,
DenseMap<DebugVariable, unsigned> &AllVarsNumbering)
{
  // mloc argument only needs the posish -> spills map and the like.
  ttracker = new TransferTracker(TII, tracker, MF, *TRI, CalleeSavedRegs);
  unsigned NumLocs = tracker->getNumLocs();

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
}

void LiveDebugValues::initial_setup(MachineFunction &MF) {
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
}

/// Calculate the liveness information for the given machine function and
/// extend ranges across basic blocks.
bool LiveDebugValues::ExtendRanges(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "\nDebug Range Extension\n");

  std::vector<MLocTransferMap> MLocTransfer;
  SmallVector<VLocTracker, 8> vlocs;
  LiveInsT SavedLiveIns;

  int MaxNumBlocks = -1;
  for (auto &MBB : MF)
    MaxNumBlocks = std::max(MBB.getNumber(), MaxNumBlocks);
  assert(MaxNumBlocks >= 0);
  ++MaxNumBlocks;

  MLocTransfer.resize(MaxNumBlocks);
  vlocs.resize(MaxNumBlocks);
  SavedLiveIns.resize(MaxNumBlocks);

  initial_setup(MF);

  produce_mloc_transfer_function(MF, MLocTransfer, MaxNumBlocks);

  // Huurrrr. Store liveouts in a massive array.
  uint64_t **MOutLocs = new uint64_t *[MaxNumBlocks];
  uint64_t **MInLocs = new uint64_t *[MaxNumBlocks];
  unsigned NumLocs = tracker->getNumLocs();
  for (int i = 0; i < MaxNumBlocks; ++i) {
    MOutLocs[i] = new uint64_t[NumLocs];
    // XXX should be zero now?
    memset(MOutLocs[i], 0xFF, sizeof(uint64_t) * NumLocs);
    MInLocs[i] = new uint64_t[NumLocs];
    memset(MInLocs[i], 0, sizeof(uint64_t) * NumLocs);
  }

  mloc_dataflow(MInLocs, MOutLocs, MLocTransfer);

  // Accumulate things into the vloc tracker.
  // Walk in RPOT order to ensure any physreg defs are seen before uses.
  for (unsigned int I = 0; I < OrderToBB.size(); ++I) {
    MachineBasicBlock &MBB = *OrderToBB[I];
    cur_bb = MBB.getNumber();
    vtracker = &vlocs[cur_bb];
    vtracker->MBB = &MBB;
    tracker->loadFromArray(MInLocs[cur_bb], cur_bb);
    cur_inst = 1;
    for (auto &MI : MBB) {
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
  for (unsigned int I = 0; I < OrderToBB.size(); ++I) {
    auto *MBB = OrderToBB[I];
    auto *vtracker = &vlocs[MBB->getNumber()];
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

  emit_locations(MF, SavedLiveIns, MInLocs, AllVarsNumbering);

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
