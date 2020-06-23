//===- LiveDebugValues.cpp - Tracking Debug Value MIs ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
/// ### Terminology
///
/// A machine location is a register or spill slot, a value is something that's
/// defined by an instruction or PHI node, while a variable value is the value
/// assigned to a variable. A variable location is a machine location, that must
/// contain the appropriate variable value. A value that is a PHI node is
/// occasionally called an mphi.
///
/// I'm calling the first dataflow problem the "machine value location" problem,
/// because we're determining which machine locations contain which values.
/// The "locations" are constant: what's unknown is what value they contain.
///
/// The second dataflow problem (the one for variables) is the "variable value
/// problem", because it's determining what values a variable has, rather than
/// what location those values are placed in. Unfortunately, it's not that
/// simple, because producing a PHI value always involves picking a location.
/// This is an imperfection that we just have to accept, IMO.
///
/// TODO:
///   Overlapping fragments
///   Entry values
///   Add back DEBUG statements for debugging this
///   Collect statistics
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
#include "llvm/Transforms/Utils/SSAUpdaterImpl.h"
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
static cl::opt<bool> EmulateOldLDV("word-wrap-like-word97", cl::Hidden,
                                   cl::desc("Act like old LiveDebugValues did"),
                                   cl::init(false));

// Rely on isStoreToStackSlotPostFE and similar to observe all stack spills.
static cl::opt<bool>
    ObserveAllStackops("observe-all-stack-ops", cl::Hidden,
                       cl::desc("Allow non-kill spill and restores"),
                       cl::init(true));

namespace {

// The location at which a spilled value resides. It consists of a register and
// an offset.
struct SpillLoc {
  unsigned SpillBase;
  int SpillOffset;
  bool operator==(const SpillLoc &Other) const {
    return SpillBase == Other.SpillBase && SpillOffset == Other.SpillOffset;
  }
  bool operator<(const SpillLoc &Other) const {
    return std::tie(SpillBase, SpillOffset) <
           std::tie(Other.SpillBase, Other.SpillOffset);
  }
};

// This is purely a number that's slightly more strongly typed, to avoid
// passing around raw integers. Identifies a register or spill slot,
// numerically.
enum LocIdx { limin = 0, limax = UINT_MAX };

#define NUM_LOC_BITS 24

/// Unique identifier for a value defined by an instruction, as a value type.
/// Casts back and forth to a uint64_t. Probably replacable with something less
/// bit-constrained. Each value identifies the instruction and machine location
/// where the value is defined, although there may be no corresponding machine
/// operand for it (ex: regmasks clobbering values). The instructions are
/// one-based, and definitions that are PHIs have instruction number zero.
///
/// The obvious limits of a 1M block function or 1M instruction blocks are
/// problematic; but by that point we should probably have bailed out of
/// trying to analyse the function.
class ValueIDNum {
public:
  uint64_t BlockNo : 20; /// The block where the def happens.
  uint64_t InstNo : 20;  /// The Instruction where the def happens.
                         /// One based, is distance from start of block.
  LocIdx LocNo : NUM_LOC_BITS; /// The machine location where the def happens.
  // (No idea why this can work as a LocIdx, it probably shouldn't)

  uint64_t asU64() const {
    uint64_t TmpBlock = BlockNo;
    uint64_t TmpInst = InstNo;
    return TmpBlock << 44ull | TmpInst << NUM_LOC_BITS | LocNo;
  }

  static ValueIDNum fromU64(uint64_t v) {
    LocIdx L = LocIdx(v & 0x3FFF);
    return {v >> 44ull, ((v >> NUM_LOC_BITS) & 0xFFFFF), L};
  }

  bool operator<(const ValueIDNum &Other) const {
    return asU64() < Other.asU64();
  }

  bool operator==(const ValueIDNum &Other) const {
    return std::tie(BlockNo, InstNo, LocNo) ==
           std::tie(Other.BlockNo, Other.InstNo, Other.LocNo);
  }

  bool operator!=(const ValueIDNum &Other) const { return !(*this == Other); }

  std::string asString(const std::string &mlocname) const {
    return Twine("bb ")
        .concat(Twine(BlockNo).concat(Twine(" inst ").concat(
            Twine(InstNo).concat(Twine(" loc ").concat(Twine(mlocname))))))
        .str();
  }
};

} // end anonymous namespace

// Boilerplate densemapinfo for ValueIDNum.
namespace llvm {
template <> struct DenseMapInfo<ValueIDNum> {
  // NB, there's a risk of overlap of uint64_max with legitmate numbering if
  // there are very many machine locations. Fix in the future by not bit packing
  // so hard.
  static const uint64_t MaxVal = std::numeric_limits<uint64_t>::max();

  static inline ValueIDNum getEmptyKey() { return ValueIDNum::fromU64(MaxVal); }

  static inline ValueIDNum getTombstoneKey() {
    return ValueIDNum::fromU64(MaxVal - 1);
  }

  static unsigned getHashValue(ValueIDNum num) {
    return hash_value(num.asU64());
  }

  static bool isEqual(const ValueIDNum &A, const ValueIDNum &B) {
    return A == B;
  }
};

// Boilerplate for our stronger-integer type.
template <> struct DenseMapInfo<LocIdx> {
  static const int MaxVal = std::numeric_limits<int>::max();

  static inline LocIdx getEmptyKey() { return LocIdx(MaxVal); }

  static inline LocIdx getTombstoneKey() { return LocIdx(MaxVal - 1); }

  static unsigned getHashValue(LocIdx Num) { return hash_value((unsigned)Num); }

  static bool isEqual(LocIdx A, LocIdx B) { return A == B; }
};

} // end namespace llvm

namespace {

/// Meta qualifiers for a value. Pair of whatever expression is used to qualify
/// the the value, and Boolean of whether or not it's indirect.
typedef std::pair<const DIExpression *, bool> MetaVal;

/// Tracker for what values are in machine locations. Listens to the Things
/// being Done by various instructions, and maintains a table of what machine
/// locations have what values (as defined by a ValueIDNum).
///
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
  const TargetFrameLowering &TFL;
  const MachineFrameInfo &MFI;

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

  /// Unique-ification of spill slots. Used to number them -- their LocID
  /// number is the index in SpillLocs minus one plus NumRegs.
  UniqueVector<SpillLoc> SpillLocs;

  // If we discover a new machine location, assign it an mphi with this
  // block number.
  unsigned CurBB;

  /// Cached local copy of the number of registers the target has.
  unsigned NumRegs;

  /// Collection of register mask operands that have been observed. Second part
  /// of pair indicates the instruction that they happened in. Used to
  /// reconstruct where defs happened if we start tracking a location later
  /// on.
  SmallVector<std::pair<const MachineOperand *, unsigned>, 32> Masks;

  MLocTracker(MachineFunction &MF, const TargetInstrInfo &TII,
              const TargetRegisterInfo &TRI, const TargetLowering &TLI,
	      const TargetFrameLowering &TFL, const MachineFrameInfo &MFI)
      : MF(MF), TII(TII), TRI(TRI), TLI(TLI), TFL(TFL), MFI(MFI) {
    NumRegs = TRI.getNumRegs();
    reset();
    LocIdxToIDNum.push_back({0, 0, LocIdx(0)});
    LocIDToLocIdx.resize(NumRegs);
    memset(&LocIDToLocIdx[0], 0, NumRegs * sizeof(LocIdx));
    LocIDToLocIdx[0] = LocIdx(0);
    LocIdxToLocID[LocIdx(0)] = 0;
    assert(NumRegs < (1u << NUM_LOC_BITS)); // Detect bit packing failure
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

  unsigned getNumLocs(void) const { return LocIdxToIDNum.size(); }

  /// Reset all locations to contain a PHI value at the designated block. Used
  /// sometimes for actual PHI values, othertimes to indicate the block entry
  /// value (before any more information is known).
  void setMPhis(unsigned NewCurBB) {
    CurBB = NewCurBB;
    for (unsigned ID = 1; ID < LocIdxToIDNum.size(); ++ID) {
      LocIdxToIDNum[LocIdx(ID)] = {CurBB, 0, LocIdx(ID)};
    }
  }

  /// Load values for each location from array of ValueIDNums. Take current
  /// bbnum just in case we read a value from a hitherto untouched register.
  void loadFromArray(uint64_t *Locs, unsigned NewCurBB) {
    CurBB = NewCurBB;
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
    // SpillsToMLocs.reset(); XXX can't reset?
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
      ValueIDNum ValNum = {CurBB, 0, NewIdx};
      // Was this reg ever touched by a regmask?
      for (auto Rit = Masks.rbegin(); Rit != Masks.rend(); ++Rit) {
        if (Rit->first->clobbersPhysReg(ID)) {
          // There was an earlier def we skipped.
          ValNum = {CurBB, Rit->second, NewIdx};
          break;
        }
      }

      LocIdxToIDNum.push_back(ValNum);
      LocIdxToLocID[NewIdx] = ID;
    }
  }

  /// Record a definition of the specified register at the given block / inst.
  /// This doesn't take a ValueIDNum, because the definition and its location
  /// are synonymous.
  void defReg(Register R, unsigned BB, unsigned Inst) {
    unsigned ID = getLocID(R, false);
    LocIdx &Idx = LocIDToLocIdx[ID];
    bumpRegister(ID, Idx);
    ValueIDNum ValueID = {BB, Inst, Idx};
    LocIdxToIDNum[Idx] = ValueID;
  }

  /// Set a register to a value number. To be used if the value number is
  /// known in advance.
  void setReg(Register R, ValueIDNum ValueID) {
    unsigned ID = getLocID(R, false);
    LocIdx &Idx = LocIDToLocIdx[ID];
    bumpRegister(ID, Idx);
    LocIdxToIDNum[Idx] = ValueID;
  }

  ValueIDNum readReg(Register R) {
    unsigned ID = getLocID(R, false);
    LocIdx &Idx = LocIDToLocIdx[ID];
    bumpRegister(ID, Idx);
    return LocIdxToIDNum[Idx];
  }

  /// Reset a register value to zero / empty. Needed to replicate old
  /// LiveDebugValues where a copy to/from a register effectively clears the
  /// contents of the source register. (Values can only have one machine
  /// location in old LiveDebugValues).
  void wipeRegister(Register R) {
    unsigned ID = getLocID(R, false);
    LocIdx Idx = LocIDToLocIdx[ID];
    LocIdxToIDNum[Idx] = {0, 0, LocIdx(0)};
  }

  /// Determine the LocIdx of an existing register.
  LocIdx getRegMLoc(Register R) {
    unsigned ID = getLocID(R, false);
    return LocIDToLocIdx[ID];
  }

  /// Record a RegMask operand being executed. Defs any register we currently
  /// track, stores a pointer to the mask in case we have to account for it
  /// later.
  void writeRegMask(const MachineOperand *MO, unsigned CurBB, unsigned InstID) {
    // Ensure SP exists, so that we don't override it later.
    unsigned SP = TLI.getStackPointerRegisterToSaveRestore();
    unsigned ID = getLocID(SP, false);
    LocIdx &Idx = LocIDToLocIdx[ID];
    bumpRegister(ID, Idx);

    // Def anything we already have that isn't preserved.
    for (auto &P : LocIdxToLocID) {
      // Don't clobber SP, even if the mask says it's clobbered.
      if (P.second != 0 && P.second < NumRegs && P.second != SP &&
          MO->clobbersPhysReg(P.second))
        defReg(P.second, CurBB, InstID);
    }
    Masks.push_back(std::make_pair(MO, InstID));
  }

  /// Set the value stored in a spill slot.
  void setSpill(SpillLoc L, ValueIDNum ValueID) {
    unsigned SpillID = SpillLocs.idFor(L);
    if (SpillID == 0) {
      SpillID = SpillLocs.insert(L);
      LocIDToLocIdx.push_back(LocIdx(0));
      unsigned L = getLocID(SpillID, true);
      LocIdx Idx = LocIdx(LocIdxToIDNum.size()); // New idx
      LocIDToLocIdx[L] = Idx;
      LocIdxToLocID[Idx] = L;
      LocIdxToIDNum.push_back(ValueID);
      assert(Idx < (1u << NUM_LOC_BITS));
    } else {
      unsigned L = getLocID(SpillID, true);
      LocIdx Idx = LocIDToLocIdx[L];
      LocIdxToIDNum[Idx] = ValueID;
    }
  }

  /// Read whatever value is in a spill slot, or zero if it isn't tracked.
  ValueIDNum readSpill(SpillLoc L) {
    unsigned SpillID = SpillLocs.idFor(L);
    if (SpillID == 0)
      // Returning no location -> $noreg, no value.
      return {0, 0, LocIdx(0)};

    unsigned LocID = getLocID(SpillID, true);
    unsigned LocIdx = LocIDToLocIdx[LocID];
    return LocIdxToIDNum[LocIdx];
  }

  /// Determine the LocIdx of a spill slot.
  LocIdx getSpillMLoc(SpillLoc L) {
    unsigned SpillID = SpillLocs.idFor(L);
    if (SpillID == 0)
      return LocIdx(0);
    unsigned LocNo = getLocID(SpillID, true);
    return LocIDToLocIdx[LocNo];
  }

  /// Return true if Idx is a spill machine location.
  bool isSpill(LocIdx Idx) const {
    auto IDIt = LocIdxToLocID.find(Idx);
    assert(IDIt != LocIdxToLocID.end());
    return IDIt->second >= NumRegs;
  }

  LocIdx MOToLocIdx(const MachineOperand &MO) {
    assert(MO.isReg() || MO.isFI());
    if (MO.isReg()) {
      Register r = MO.getReg();
      if (r == 0)
        return LocIdx(0);
      return LocIDToLocIdx[r]; // possibly 0
    } else {
      unsigned FI = MO.getIndex();
      if (!MFI.isDeadObjectIndex(FI)) {
        unsigned Base;
        int64_t offs = TFL.getFrameIndexReference(MF, FI, Base);
        SpillLoc SL = {Base, (int)offs}; // XXX loss of 64 to 32?
        return getSpillMLoc(SL);
      } else {
        return LocIdx(0);
      }
    }
  }

  std::string LocIdxToName(LocIdx Idx) const {
    auto IDIt = LocIdxToLocID.find(Idx);
    assert(IDIt != LocIdxToLocID.end());
    unsigned ID = IDIt->second;
    if (ID >= NumRegs)
      return Twine("slot ").concat(Twine(ID - NumRegs)).str();
    else
      return TRI.getRegAsmName(ID).str();
  }

  std::string IDAsString(const ValueIDNum &Num) const {
    std::string DefName = LocIdxToName(Num.LocNo);
    return Num.asString(DefName);
  }

  LLVM_DUMP_METHOD
  void dump() const {
    for (unsigned int ID = 0; ID < LocIdxToIDNum.size(); ++ID) {
      auto &ValueID = LocIdxToIDNum[ID];
      if (ValueID.LocNo == 0)
        continue;
      std::string MLocName = LocIdxToName(ValueID.LocNo);
      std::string DefName = ValueID.asString(MLocName);
      dbgs() << LocIdxToName(LocIdx(ID)) << " --> " << DefName << "\n";
    }
  }

  LLVM_DUMP_METHOD
  void dump_mloc_map() const {
    for (unsigned I = 0; I < LocIdxToIDNum.size(); ++I) {
      std::string foo = LocIdxToName(LocIdx(I));
      dbgs() << "Idx " << I << " " << foo << "\n";
    }
  }

  /// Create a DBG_VALUE based on  machine location \p MLoc. Qualify it with the
  /// information in meta, for variable Var. Don't insert it anywhere, just
  /// return the builder for it.
  MachineInstrBuilder emitLoc(LocIdx MLoc, const DebugVariable &Var,
                              const MetaVal &Meta) {
    DebugLoc DL =
        DebugLoc::get(0, 0, Var.getVariable()->getScope(), Var.getInlinedAt());
    auto MIB = BuildMI(MF, DL, TII.get(TargetOpcode::DBG_VALUE));

    const DIExpression *Expr = Meta.first;
    unsigned Loc = LocIdxToLocID[MLoc];
    if (Loc >= NumRegs) {
      const SpillLoc &Spill = SpillLocs[Loc - NumRegs + 1];
      Expr = DIExpression::prepend(Expr, DIExpression::ApplyOffset,
                                   Spill.SpillOffset);
      unsigned Base = Spill.SpillBase;
      MIB.addReg(Base, RegState::Debug);
      MIB.addImm(0);
    } else {
      MIB.addReg(Loc, RegState::Debug);
      if (Meta.second)
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

  bool operator!=(const ValueRec &Other) const { return !(*this == Other); }
};

/// Types for recording sets of variable fragments that overlap. For a given
/// local variable, we record all other fragments of that variable that could
/// overlap it, to reduce search time.
using FragmentOfVar =
    std::pair<const DILocalVariable *, DIExpression::FragmentInfo>;
using OverlapMap =
    DenseMap<FragmentOfVar, SmallVector<DIExpression::FragmentInfo, 1>>;

/// Collection of DBG_VALUEs observed when traversing a block. Records each
/// variable and the value the DBG_VALUE refers to. Requires the machine value
/// location dataflow algorithm to have run already, so that values can be
/// identified.
class VLocTracker {
public:
  /// Map DebugVariable to the latest Value it's defined to have.
  /// Needs to be a mapvector because we determine order-in-the-input-MIR from
  /// the order in this thing.
  MapVector<DebugVariable, ValueRec> Vars;
  DenseMap<DebugVariable, const DILocation *> Scopes;
  MachineBasicBlock *MBB;

public:
  VLocTracker() {}

  void defVar(const MachineInstr &MI, ValueIDNum ID) {
    // XXX skipping overlapping fragments for now.
    assert(MI.isDebugValue() || MI.isDebugRef());
    DebugVariable Var(MI.getDebugVariable(), MI.getDebugExpression(),
                      MI.getDebugLoc()->getInlinedAt());
    MetaVal Meta = {MI.getDebugExpression(), MI.getOperand(1).isImm()};
    Vars[Var] = {ID, None, Meta, ValueRec::Def};
    Scopes[Var] = MI.getDebugLoc().get();
  }

  void defVar(const MachineInstr &MI, const MachineOperand &MO) {
    // XXX skipping overlapping fragments for now.
    assert(MI.isDebugValue() || MI.isDebugRef());
    DebugVariable Var(MI.getDebugVariable(), MI.getDebugExpression(),
                      MI.getDebugLoc()->getInlinedAt());
    MetaVal Meta = {MI.getDebugExpression(), MI.getOperand(1).isImm()};
    Vars[Var] = {{0, 0, LocIdx(0)}, MO, Meta, ValueRec::Const};
    Scopes[Var] = MI.getDebugLoc().get();
  }
};

/// Tracker for converting machine value locations and variable values into
/// variable locations (the output of LiveDebugValues), recorded as DBG_VALUEs
/// specifying block live-in locations and transfers within blocks.
///
/// Operating on a per-block basis, this class takes a (pre-loaded) MLocTracker
/// and must be initialized with the set of variable values that are live-in to
/// the block. The caller then repeatedly calls process(). TransferTracker picks
/// out variable locations for the live-in variable values (if there _is_ a
/// location) and creates the corresponding DBG_VALUEs. Then, as the block is
/// stepped through, transfers of values between machine locations are
/// identified and if profitable, a DBG_VALUE created.
///
/// This is where debug use-before-defs would be resolved: a variable with an
/// unavailable value could materialize in the middle of a block, when the
/// value becomes available. Or, we could detect clobbers and re-specify the
/// variable in a backup location. (XXX these are unimplemented).
class TransferTracker {
public:
  const TargetInstrInfo *TII;
  /// This machine location tracker is assumed to always contain the up-to-date
  /// value mapping for all machine locations. TransferTracker only reads
  /// information from it. (XXX make it const?)
  MLocTracker *MTracker;
  MachineFunction &MF;

  /// Record of all changes in variable locations at a block position. Awkwardly
  /// we allow inserting either before or after the point: MBB != nullptr
  /// indicates it's before, otherwise after.
  struct Transfer {
    MachineBasicBlock::iterator Pos;   /// Position to insert DBG_VALUes
    MachineBasicBlock *MBB;            /// non-null if we should insert after.
    SmallVector<MachineInstr *, 4> Insts; /// Vector of DBG_VALUEs to insert.
  };

  typedef std::pair<LocIdx, MetaVal> LocAndMeta;

  /// Collection of transfers (DBG_VALUEs) to be inserted.
  SmallVector<Transfer, 32> Transfers;

  /// Local cache of what-value-is-in-what-LocIdx. Used to identify differences
  /// between TransferTrackers view of variable locations and MLocTrackers. For
  /// example, MLocTracker observes all clobbers, but TransferTracker lazily
  /// does not.
  std::vector<ValueIDNum> VarLocs;

  /// Map from LocIdxes to which DebugVariables are based that location.
  /// Mantained while stepping through the block. Not accurate if
  /// VarLocs[Idx] != MTracker->LocIdxToIDNum[Idx].
  DenseMap<LocIdx, SmallSet<DebugVariable, 4>> ActiveMLocs;

  /// Map from DebugVariable to it's current location and qualifying meta
  /// information. To be used in conjunction with ActiveMLocs to construct
  /// enough information for the DBG_VALUEs for a particular LocIdx.
  DenseMap<DebugVariable, LocAndMeta> ActiveVLocs;

  /// Temporary cache of DBG_VALUEs to be entered into the Transfers collection.
  SmallVector<MachineInstr *, 4> PendingDbgValues;

  const TargetRegisterInfo &TRI;
  const BitVector &CalleeSavedRegs;

  class UseBeforeDef {
  public:
    ValueIDNum ID;
    DebugVariable Var;
    MetaVal m;
  };
  std::map<unsigned, SmallVector<UseBeforeDef, 1>> UseBeforeDefs;

  TransferTracker(const TargetInstrInfo *TII, MLocTracker *MTracker,
                  MachineFunction &MF, const TargetRegisterInfo &TRI,
                  const BitVector &CalleeSavedRegs)
      : TII(TII), MTracker(MTracker), MF(MF), TRI(TRI),
        CalleeSavedRegs(CalleeSavedRegs) {}

  /// Load object with live-in variable values. \p mlocs contains the live-in
  /// values in each machine location, while \p vlocs the live-in variable
  /// values. This method picks variable locations for the live-in variables,
  /// creates DBG_VALUEs and puts them in #Transfers, then prepares the other
  /// object fields to track variable locations as we step through the block.
  /// FIXME: could just examine mloctracker instead of passing in \p mlocs?
  void loadInlocs(MachineBasicBlock &MBB, uint64_t *MLocs,
                  SmallVectorImpl<std::pair<DebugVariable, ValueRec>> &VLocs,
                  unsigned NumLocs, unsigned CurBB) {
    ActiveMLocs.clear();
    ActiveVLocs.clear();
    VarLocs.clear();
    VarLocs.resize(NumLocs);
    UseBeforeDefs.clear();

    auto isCalleeSaved = [&](LocIdx L) {
      unsigned Reg = MTracker->LocIdxToLocID[L];
      for (MCRegAliasIterator RAI(Reg, &TRI, true); RAI.isValid(); ++RAI)
        if (CalleeSavedRegs.test(*RAI))
          return true;
      return false;
    };

    // Map of the preferred location for each value.
    DenseMap<ValueIDNum, LocIdx> ValueToLoc;

    // Produce a map of value numbers to the current machine locs they live
    // in. When emulating old LiveDebugValues, there should only be one
    // location; when not, we get to pick.
    for (unsigned Idx = 1; Idx < NumLocs; ++Idx) {
      auto VNum = ValueIDNum::fromU64(MLocs[Idx]);
      VarLocs[Idx] = VNum;
      auto it = ValueToLoc.find(VNum);
      // If there's no location for this value yet; or it's a spill, or not a
      /// preferred non-volatile register, then pick this location.
      if (it == ValueToLoc.end() || MTracker->isSpill(it->second) ||
          !isCalleeSaved(it->second))
        ValueToLoc[VNum] = LocIdx(Idx);
    }

    // Now map variables to their picked LocIdxes.
    for (auto Var : VLocs) {
      if (Var.second.Kind == ValueRec::Const) {
        PendingDbgValues.push_back(
            emitMOLoc(*Var.second.MO, Var.first, Var.second.meta));
        continue;
      }

      // If the value has no location, we can't make a variable location.
      // Test for use before defs first.
      auto ValuesPreferredLoc = ValueToLoc.find(Var.second.ID);
      if (ValuesPreferredLoc == ValueToLoc.end()) {
        if (Var.second.ID.BlockNo != CurBB || Var.second.ID.InstNo == 0)
          continue;
        // Otherwise it's in this block and not a PHI.
        UseBeforeDefs[Var.second.ID.InstNo].push_back(
                 UseBeforeDef{Var.second.ID, Var.first, Var.second.meta});
        continue;
      }

      LocIdx M = ValuesPreferredLoc->second;
      ActiveVLocs[Var.first] = std::make_pair(M, Var.second.meta);
      ActiveMLocs[M].insert(Var.first);
      assert(M != 0);
      PendingDbgValues.push_back(
          MTracker->emitLoc(M, Var.first, Var.second.meta));
    }
    flushDbgValues(MBB.begin(), &MBB);
  }

  void prodAfterInst(unsigned inst, MachineBasicBlock::iterator pos) {
    auto mit = UseBeforeDefs.find(inst);
    if (mit == UseBeforeDefs.end())
      return;

    for (auto &Use : mit->second) {
      LocIdx L = Use.ID.LocNo;

      // Problem: we can name defs on copies that we later look through. This
      // means there are values we can't actually track. This sucks.
      if (MTracker->LocIdxToIDNum[L] != Use.ID)
        continue;

      PendingDbgValues.push_back(MTracker->emitLoc(L, Use.Var, Use.m));
    }
    flushDbgValues(pos, nullptr);
  }

  /// Helper to move created DBG_VALUEs into Transfers collection.
  void flushDbgValues(MachineBasicBlock::iterator Pos, MachineBasicBlock *MBB) {
    if (PendingDbgValues.size() > 0) {
      Transfers.push_back({Pos, MBB, PendingDbgValues});
      PendingDbgValues.clear();
    }
  }

  /// Handle a DBG_VALUE within a block. Terminate the variables current
  /// location, and record the value its DBG_VALUE refers to, so that we can
  /// detect location transfers later on.
  void redefVar(const MachineInstr &MI, LocIdx NewLoc = LocIdx(0)) {
    DebugVariable Var(MI.getDebugVariable(), MI.getDebugExpression(),
                      MI.getDebugLoc()->getInlinedAt());

    // Erase any previous location,
    auto It = ActiveVLocs.find(Var);
    if (It != ActiveVLocs.end()) {
      ActiveMLocs[It->second.first].erase(Var);
    }

    // Insert a new variable location. Ignore non-register locations, we don't
    // transfer those, and can't currently describe spill locs independently of
    // regs.
    // (This is because a spill location is a DBG_VALUE of the stack pointer).
    const MachineOperand &MO = MI.getOperand(0);
    if (NewLoc == 0 && (!MO.isReg() || MO.getReg() == 0)) {
      if (It != ActiveVLocs.end())
        ActiveVLocs.erase(It);
      return;
    }

    if (NewLoc == 0) {
      Register Reg = MO.getReg();
      NewLoc = MTracker->getRegMLoc(Reg);
    }
    MetaVal Meta = {MI.getDebugExpression(), MI.getOperand(1).isImm()};

    // Check whether our local copy of values-by-location in #VarLocs is out of
    // date. Wipe old tracking data for the location if it's been clobbered in
    // the meantime.
    if (MTracker->getNumAtPos(NewLoc) != VarLocs[NewLoc]) {
      for (auto &P : ActiveMLocs[NewLoc]) {
        ActiveVLocs.erase(P);
      }
      ActiveMLocs[NewLoc].clear();
      VarLocs[NewLoc] = MTracker->getNumAtPos(NewLoc);
    }

    ActiveMLocs[NewLoc].insert(Var);
    if (It == ActiveVLocs.end()) {
      ActiveVLocs.insert(std::make_pair(Var, std::make_pair(NewLoc, Meta)));
    } else {
      It->second.first = NewLoc;
      It->second.second = Meta;
    }
  }

  /// Explicitly terminate variable locations based on \p mloc. Creates undef
  /// DBG_VALUEs for any variables that were located there, and clears
  /// #ActiveMLoc / #ActiveVLoc tracking information for that location.
  void clobberMloc(LocIdx MLoc, MachineBasicBlock::iterator Pos) {
    assert(MTracker->isSpill(MLoc));
    auto ActiveMLocIt = ActiveMLocs.find(MLoc);
    if (ActiveMLocIt == ActiveMLocs.end())
      return;

    VarLocs[MLoc] = ValueIDNum{0, 0, LocIdx(0)};

    for (auto &Var : ActiveMLocIt->second) {
      auto ActiveVLocIt = ActiveVLocs.find(Var);
      // Create an undef. We can't feed in a nullptr DIExpression alas,
      // so use the variables last expression.
      const DIExpression *Expr = ActiveVLocIt->second.second.first;
      LocIdx Idx = LocIdx(0);
      PendingDbgValues.push_back(MTracker->emitLoc(Idx, Var, {Expr, false}));
      ActiveVLocs.erase(ActiveVLocIt);
    }
    flushDbgValues(Pos, nullptr);

    ActiveMLocIt->second.clear();
  }

  /// Transfer variables based on \p Src to be based on \p Dst. This handles
  /// both register copies as well as spills and restores. Creates DBG_VALUEs
  /// describing the movement.
  void transferMlocs(LocIdx Src, LocIdx Dst, MachineBasicBlock::iterator Pos) {
    // Does Src still contain the value num we expect? If not, it's been
    // clobbered in the meantime, and our variable locations are stale.
    if (VarLocs[Src] != MTracker->getNumAtPos(Src))
      return;

    // assert(ActiveMLocs[Dst].size() == 0);
    //^^^ Legitimate scenario on account of un-clobbered slot being assigned to?
    ActiveMLocs[Dst] = ActiveMLocs[Src];
    VarLocs[Dst] = VarLocs[Src];

    // For each variable based on Src; create a location at Dst.
    for (auto &Var : ActiveMLocs[Src]) {
      auto ActiveVLocIt = ActiveVLocs.find(Var);
      assert(ActiveVLocIt != ActiveVLocs.end());
      ActiveVLocIt->second.first = Dst;

      assert(Dst != 0);
      MachineInstr *MI =
          MTracker->emitLoc(Dst, Var, ActiveVLocIt->second.second);
      PendingDbgValues.push_back(MI);
    }
    ActiveMLocs[Src].clear();
    flushDbgValues(Pos, nullptr);

    // XXX XXX XXX "pretend to be old LDV" means dropping all tracking data
    // about the old location.
    if (EmulateOldLDV)
      VarLocs[Src] = ValueIDNum{0, 0, LocIdx(0)};
  }

  MachineInstrBuilder emitMOLoc(const MachineOperand &MO,
                                const DebugVariable &Var, const MetaVal &Meta) {
    DebugLoc DL =
        DebugLoc::get(0, 0, Var.getVariable()->getScope(), Var.getInlinedAt());
    auto MIB = BuildMI(MF, DL, TII->get(TargetOpcode::DBG_VALUE));
    MIB.add(MO);
    if (Meta.second)
      MIB.addImm(0);
    else
      MIB.addReg(0);
    MIB.addMetadata(Var.getVariable());
    MIB.addMetadata(Meta.first);
    return MIB;
  }
};

class jmorseupdater;

class LiveDebugValues : public MachineFunctionPass {
private:
  using FragmentInfo = DIExpression::FragmentInfo;
  using OptFragmentInfo = Optional<DIExpression::FragmentInfo>;

  // Helper while building OverlapMap, a map of all fragments seen for a given
  // DILocalVariable.
  using VarToFragments =
      DenseMap<const DILocalVariable *, SmallSet<FragmentInfo, 4>>;

  /// Machine location/value transfer function, a mapping of which locations
  // are assigned which new values.
  typedef DenseMap<LocIdx, ValueIDNum> MLocTransferMap;

  /// Live in/out structure for the variable values: a per-block map of
  /// variables to their values. XXX, better name?
  typedef DenseMap<const MachineBasicBlock *,
                   DenseMap<DebugVariable, ValueRec> *>
      LiveIdxT;

  typedef std::pair<DebugVariable, ValueRec> VarAndLoc;

  /// Vector (per block) of a collection (inner smallvector) of live-ins.
  /// Used as the result type for the variable value dataflow problem.
  typedef SmallVector<SmallVector<VarAndLoc, 8>, 8> LiveInsT;

  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;
  const TargetFrameLowering *TFI;
  const MachineFrameInfo *MFI;
  BitVector CalleeSavedRegs;
  LexicalScopes LS;

  /// Object to track machine locations as we step through a block. Could
  /// probably be a field rather than a pointer, as it's always used.
  MLocTracker *MTracker;

  /// Number of the current block LiveDebugValues is stepping through.
  unsigned CurBB;

  /// Number of the current instruction LiveDebugValues is evaluating.
  unsigned CurInst;

  /// Variable tracker -- listens to DBG_VALUEs occurring as LiveDebugValues
  /// steps through a block. Reads the (pre-solved) values at each location
  /// from the MLocTracker object.
  VLocTracker *VTracker;

  /// Tracker for transfers, listens to DBG_VALUEs and transfers of values
  /// between locations during stepping, creates new DBG_VALUEs when values move
  /// location.
  TransferTracker *TTracker;

  /// Blocks which are artificial, i.e. blocks which exclusively contain
  /// instructions without DebugLocs, or with line 0 locations.
  SmallPtrSet<const MachineBasicBlock *, 16> ArtificialBlocks;

  // Mapping of blocks to and from their RPOT order.
  DenseMap<unsigned int, MachineBasicBlock *> OrderToBB;
  DenseMap<MachineBasicBlock *, unsigned int> BBToOrder;
  DenseMap<unsigned, unsigned> BBNumToRPO;

  // Map of overlapping variable fragments.
  OverlapMap OverlapFragments;
  VarToFragments SeenFragments;

  typedef std::pair<MachineInstr *, unsigned> SeenInst;
  std::map<uint64_t, SeenInst> InstrIDMap; // key is inst id, not full ID

  std::map<DebugInstrRefID, ValueIDNum> abimap_regs;

  /// Tests whether this instruction is a spill to a stack slot.
  bool isSpillInstruction(const MachineInstr &MI, MachineFunction *MF);

  /// Decide if @MI is a spill instruction and return true if it is. We use 2
  /// criteria to make this decision:
  /// - Is this instruction a store to a spill slot?
  /// - Is there a register operand that is both used and killed?
  /// TODO: Store optimization can fold spills into other stores (including
  /// other spills). We do not handle this yet (more than one memory operand).
  bool isLocationSpill(const MachineInstr &MI, MachineFunction *MF,
                       unsigned &Reg);

  /// If a given instruction is identified as a spill, return the spill slot
  /// and set \p Reg to the spilled register.
  Optional<SpillLoc> isRestoreInstruction(const MachineInstr &MI,
                                          MachineFunction *MF, unsigned &Reg);

  /// Given a spill instruction, extract the register and offset used to
  /// address the spill slot in a target independent way.
  SpillLoc extractSpillBaseRegAndOffset(const MachineInstr &MI);

  /// Observe a single instruction while stepping through a block.
  void process(MachineInstr &MI, uint64_t **MInLocs = nullptr);

  /// Examines whether \p MI is a DBG_VALUE and notifies trackers. 
  /// \returns true if MI was recognized and processed.
  bool transferDebugValue(const MachineInstr &MI);

  bool transferDebugInstrRef(MachineInstr &MI, uint64_t **MInLocs);

  /// Examines whether \p MI is copy instruction, and notifies trackers.
  /// \returns true if MI was recognized and processed.
  bool transferRegisterCopy(MachineInstr &MI);

  /// Examines whether \p MI is stack spill or restore  instruction, and
  /// notifies trackers. \returns true if MI was recognized and processed.
  bool transferSpillOrRestoreInst(MachineInstr &MI);

  /// Examines \p MI for any registers that it defines, and notifies trackers.
  /// \returns true if MI was recognized and processed.
  void transferRegisterDef(MachineInstr &MI);

  /// Copy one location to the other, accounting for movement of subregisters
  /// too.
  void performCopy(Register Src, Register Dst);

  void accumulateFragmentMap(MachineInstr &MI);

  /// Step through the function, recording register definitions and movements
  /// in an MLocTracker. Convert the observations into a per-block transfer
  /// function in \p MLocTransfer, suitable for using with the machine value
  /// location dataflow problem.
  void produceMLocTransferFunction(MachineFunction &MF,
                                   SmallVectorImpl<MLocTransferMap> &MLocTransfer,
                                   unsigned MaxNumBlocks);

  /// Solve the machine value location dataflow problem. Takes as input the
  /// transfer functions in \p MLocTransfer. Writes the output live-in and
  /// live-out arrays to the (initialized to zero) multidimensional arrays in
  /// \p MInLocs and \p MOutLocs. The outer dimension is indexed by block
  /// number, the inner by LocIdx.
  void mlocDataflow(uint64_t **MInLocs, uint64_t **MOutLocs,
                    SmallVectorImpl<MLocTransferMap> &MLocTransfer);

  /// Perform a control flow join (lattice value meet) of the values in machine
  /// locations at \p MBB. Follows the algorithm described in the file-comment,
  /// reading live-outs of predecessors from \p OutLocs, the current live ins
  /// from \p InLocs, and assigning the newly computed live ins back into
  /// \p InLocs. \returns true if a change was made.
  bool mlocJoin(MachineBasicBlock &MBB,
                SmallPtrSet<const MachineBasicBlock *, 16> &Visited,
                uint64_t **OutLocs, uint64_t *InLocs);

  /// Solve the variable value dataflow problem, for a single lexical scope.
  /// Uses the algorithm from the file comment to resolve control flow joins,
  /// although there are extra hacks, see vlocJoinLocation. Reads the
  /// locations of values from the \p MInLocs and \p MOutLocs arrays (see
  /// mlocDataflow) and reads the variable values transfer function from
  /// \p AllTheVlocs. Live-in and Live-out variable values are stored locally,
  /// with the live-ins permanently stored to \p Output once the fixedpoint is
  /// reached.
  /// \p VarsWeCareAbout contains a collection of the variables in \p Scope
  /// that we should be tracking.
  /// \p AssignBlocks contains the set of blocks that aren't in \p Scope, but
  /// which do contain DBG_VALUEs, which old LiveDebugValues tracked locations
  /// through.
  void vlocDataflow(const LexicalScope *Scope, const DILocation *DILoc,
                    const SmallSet<DebugVariable, 4> &VarsWeCareAbout,
                    SmallPtrSetImpl<MachineBasicBlock *> &AssignBlocks,
                    LiveInsT &Output, uint64_t **MOutLocs, uint64_t **MInLocs,
                    SmallVectorImpl<VLocTracker> &AllTheVLocs);

  /// Compute the live-ins to a block, considering control flow merges according
  /// to the method in the file comment. Live out and live in variable values
  /// are stored in \p VLOCOutLocs and \p VLOCInLocs, while machine value
  /// locations are in \p MOutLocs and \p MInLocs. The live-ins for \p MBB are
  /// computed and stored into \p VLOCInLocs. \returns true if the live-ins
  /// are modified. Delegates most logic for merging to \ref vlocJoinLocation.
  bool vlocJoin(MachineBasicBlock &MBB, LiveIdxT &VLOCOutLocs,
                LiveIdxT &VLOCInLocs,
                SmallPtrSet<const MachineBasicBlock *, 16> *VLOCVisited,
                unsigned BBNum, const SmallSet<DebugVariable, 4> &AllVars,
                uint64_t **MInLocs, uint64_t **MOutLocs,
                SmallPtrSet<const MachineBasicBlock *, 8> &NonAssignBlocks);

  /// Perform value merge for a single variable at a particular block,
  /// between two individual predecessor values. \ref vlocJoin picks one
  /// predecessor live-out value as a "base" live-in, then merges all the
  /// other predecessor values into it with this method.
  /// \p InLoc One of the incoming values,
  /// \p OLoc The other incoming value.
  /// \p PrevInLocs Map of what the previous live-in values were.
  /// \p CurVar the DebugVariable we're working with.
  /// \p ThisIsABackEdge True if \p OLoc is a backedge.
  /// \returns true if the values are reconciled, false if there is an
  /// unresolvable value conflict.
  bool vlocJoinLocation(MachineBasicBlock &MBB, const ValueRec &InLoc,
                        const ValueRec &OLoc, uint64_t *InLocOutLocs,
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
  void emitLocations(MachineFunction &MF, LiveInsT SavedLiveIns,
                     uint64_t **MInLocs,
                     DenseMap<DebugVariable, unsigned> &AllVarsNumbering);

  /// Boilerplate computation of some initial sets, artifical blocks and
  /// RPOT block ordering.
  void initialSetup(MachineFunction &MF);

  void do_the_re_ssaifying_dance(MachineFunction &MF, uint64_t **MLiveIns, uint64_t **MLiveOuts);

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

  bool isCalleeSaved(LocIdx L) {
    unsigned Reg = MTracker->LocIdxToLocID[L];
    for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
      if (CalleeSavedRegs.test(*RAI))
        return true;
    return false;
  }

  // Uhhhhhhh
  std::map<uint64_t, std::set<MachineInstr *>> SeenInstrIDs;
  std::map<uint64_t, std::pair<MachineInstr *, std::vector<ValueIDNum>>> DebugReadPoints;
  friend class SSAUpdaterTraits<jmorseupdater>;
  std::map<MachineInstr *, ValueIDNum> dephid_instr_resolutions;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
//            Implementation
//===----------------------------------------------------------------------===//

char LiveDebugValues::ID = 0;

char &llvm::LiveDebugValuesID = LiveDebugValues::ID;

INITIALIZE_PASS(LiveDebugValues, DEBUG_TYPE, "Live DEBUG_VALUE analysis", false,
                false)

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
// void LiveDebugValues::printVarLocInMBB(..)
#endif

SpillLoc LiveDebugValues::extractSpillBaseRegAndOffset(const MachineInstr &MI) {
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
  // at all, this variable shouldn't get a legitimate location range.
  auto *Scope = LS.findLexicalScope(MI.getDebugLoc().get());
  if (Scope == nullptr)
    return true; // handled it; by doing nothing

  const MachineOperand &MO = MI.getOperand(0);

  // MLocTracker needs to know that this register is read, even if it's only
  // read by a debug inst.
  if (MO.isReg() && MO.getReg() != 0)
    (void)MTracker->readReg(MO.getReg());

  // If we're preparing for the second analysis (variables), the machine value
  // locations are already solved, and we report this DBG_VALUE and the value
  // it refers to to VLocTracker.
  if (VTracker) {
    if (MO.isReg()) {
      // Should read LocNo==0 on $noreg.
      ValueIDNum Undef = {0, 0, LocIdx(0)};
      ValueIDNum ID = (MO.getReg()) ? MTracker->readReg(MO.getReg()) : Undef;
      VTracker->defVar(MI, ID);
    } else if (MI.getOperand(0).isImm() || MI.getOperand(0).isFPImm() ||
               MI.getOperand(0).isCImm()) {
      VTracker->defVar(MI, MI.getOperand(0));
    }
  }

  // If performing final tracking of transfers, report this variable definition
  // to the TransferTracker too.
  if (TTracker)
    TTracker->redefVar(MI);
  return true;
}

bool LiveDebugValues::transferDebugInstrRef(MachineInstr &MI, uint64_t **MInLocs)
{
  if (!MI.isDebugRef())
    return false;

  if (!VTracker) {
    auto ID = DebugInstrRefID::fromU64(MI.getOperand(0).getImm());
    // XXX XXX XXX that ID might need updating.
    SeenInstrIDs[ID.getInstID()].insert(&MI);;
    return false;
  }

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

  const MachineFunction &MF = *MI.getParent()->getParent();

  // Do we have a location for the designated value ID?
  const MachineOperand &MO = MI.getOperand(0);
  assert(MO.isImm());
  auto ID = DebugInstrRefID::fromU64(MO.getImm());

  // It might be subject to some kind of update...
  auto It = MF.valueIDUpdateMap.find(ID);
  unsigned SubReg = 0;
  while (It != MF.valueIDUpdateMap.end()) {
    ID = It->second.first;
    SubReg = (SubReg) ? SubReg : It->second.second; // Pick out first subreg
    // Should be the smallest; and any coalescing in the meantime can't have
    // moved it.
    It = MF.valueIDUpdateMap.find(ID);
  }

  ValueIDNum NewID;

  // Is it a.... PHI?
  auto PHIIt = MF.PHIPointToReg.find(ID);
  auto InstrIt = InstrIDMap.find(ID.getInstID());
  auto dit = dephid_instr_resolutions.find(&MI);
  if (PHIIt != MF.PHIPointToReg.end()) {
    // Only handle reg phis for now...
    LocIdx L = LocIdx(0);
    if (PHIIt->second.second.isReg()) {
      unsigned LocID = MTracker->getLocID(PHIIt->second.second.getReg(), false);
      L = MTracker->LocIDToLocIdx[LocID];
      SubReg = PHIIt->second.second.getSubReg();
    } else {
      assert(PHIIt->second.second.isFI());
      unsigned FI = PHIIt->second.second.getIndex();
      unsigned Base;
      if (!MFI->isDeadObjectIndex(FI)) {
        int64_t offs = TFI->getFrameIndexReference(MF, FI, Base);
        SpillLoc SL = {Base, (int)offs}; // XXX loss of 64 to 32?
        L = MTracker->getSpillMLoc(SL);
      } else {
        // It's dead jim.
        ;
      }
    }

    // We could directly produce an mphi value here, however, branch folding
    // can shuffle branches to the point where the phi-ness has gone away
    // and an explicit location is known. Pick the location out of MInLocs.
    // Technically after this point some pass could start rewriting registers
    // too, but nothing does that right now.
    NewID = ValueIDNum::fromU64(MInLocs[PHIIt->second.first->getNumber()][L]);
  } else if (InstrIt != InstrIDMap.end()) {
    // No: it must refer to an instruction.
    if (InstrIt != InstrIDMap.end()) {
      uint64_t BlockNo = InstrIt->second.first->getParent()->getNumber();
      if (ID.isOperand()) { 
        const MachineOperand &MO = InstrIt->second.first->getOperand(ID.getOperand());
        assert(MO.isReg());
        unsigned LocID = MTracker->getLocID(MO.getReg(), false);
        LocIdx L = MTracker->LocIDToLocIdx[LocID]; // might be 0
        NewID = ValueIDNum{BlockNo, InstrIt->second.second, L};
      } else {
        // This is a physreg at a particular position. Is it in the abi regs map?
        auto it = abimap_regs.find(ID);
        if (it != abimap_regs.end())
          NewID = it->second;
        // else: it's a use-before-def. Technically we can avoid this., but
        // XXX XXX XXX skip it for now.
      }
    } else {
      // No observed instrs: it's optimised out.
      NewID = ValueIDNum{0, 0, LocIdx(0)};
    }
  } else if (dit != dephid_instr_resolutions.end()) {
    NewID = dit->second;
  } else {
    ; // it is no-where.
  }

  // We picked up a subregister along the way; check whether the def at this
  // location should actually refer to one of its subregisters.
  if (SubReg != 0) {
    LocIdx l = NewID.LocNo;
    unsigned ID = MTracker->LocIdxToLocID[l];
    if (ID < MTracker->NumRegs) {
      // Is this register already in that calss?
      unsigned res = TRI->getSubReg(ID, SubReg);
      if (res != 0) {
        //assert(MTracker->LocIDToLocIdx[res] != 0); // tooottallly going to fail
        if (MTracker->LocIDToLocIdx[res] != 0) // tooottallly going to fail
          NewID.LocNo = MTracker->LocIDToLocIdx[res];
        else
          NewID.LocNo = LocIdx(0); // This happens once in clang-3.4. Needs study
      }
      // XXX need to assert that any un-used subreg is because the old loc
      // and the subreg loc are the same size. Probably means carrying around
      // a register class pointer.
    } else {
      // We can land in a stack slot. This is, in theory, fine, and we can
      // do stuff about that. However, I'm now increadibly bored, and thus
      // will leave this dangling for the moment
      // XXX XXX XXX           XXX XXX XXX
      // XXX XXX XXX           XXX XXX XXX
      // XXX XXX XXX           XXX XXX XXX
      //            XXX XXX XXX
      //            XXX XXX XXX
      //            XXX XXX XXX
      // XXX XXX XXX           XXX XXX XXX
      // XXX XXX XXX           XXX XXX XXX
      // XXX XXX XXX           XXX XXX XXX
      SubReg = 0;
    }
  }

  // OK, we have a Value ID. Set it.
  VTracker->defVar(MI, NewID);

  if (TTracker) {
    // Is that value tracked anywhere right now? XXX XXX XXX speed.
    LocIdx L = LocIdx(0);
    for (unsigned int I = 0; I < MTracker->LocIdxToIDNum.size(); ++I) {
      LocIdx CurL = LocIdx(I);
      ValueIDNum ID = MTracker->LocIdxToIDNum[CurL];
      if (ID == NewID) {
        unsigned LID = MTracker->LocIdxToLocID[L];
        unsigned CurLID = MTracker->LocIdxToLocID[LocIdx(CurL)];
        if (L != 0 && CurL >= MTracker->NumRegs)
          L = CurL; // override spills
        else if (L != 0 && LID < MTracker->NumRegs && CurLID < MTracker->NumRegs && !isCalleeSaved(L) && isCalleeSaved(CurL))
          L = CurL; // override volatiles
        else if (L == 0)
          L = CurL; // override empty
      }
    }
    TTracker->redefVar(MI, L);

    // Also drop out a DBG_VALUE!
    MetaVal Meta = {MI.getDebugExpression(), MI.getOperand(1).isImm()};
    MachineInstr *DbgMI = MTracker->emitLoc(L, V, Meta);
    TTracker->PendingDbgValues.push_back(DbgMI);
    TTracker->flushDbgValues(MI.getIterator(), nullptr);
  }

  return true;
}

void LiveDebugValues::transferRegisterDef(MachineInstr &MI) {
  // Meta Instructions do not affect the debug liveness of any register they
  // define.
  if (MI.isImplicitDef()) {
    // Except when there's an implicit def, and the location it's defining has
    // no value number. The whole point of an implicit def is to announce that
    // the register is live, without be specific about it's value. So define
    // a value if there isn't one already.
    ValueIDNum Num = MTracker->readReg(MI.getOperand(0).getReg());
    // Has a legitimate value -> ignore the implicit def.
    if (Num.LocNo != 0)
      return;
    // Otherwise, def it here.
  } else if (MI.isMetaInstruction())
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
    MTracker->defReg(DeadReg, CurBB, CurInst);

  for (auto *MO : RegMaskPtrs)
    MTracker->writeRegMask(MO, CurBB, CurInst);
}


void LiveDebugValues::performCopy(Register SrcRegNum, Register DstRegNum) {
  ValueIDNum SrcValue = MTracker->readReg(SrcRegNum);

  MTracker->setReg(DstRegNum, SrcValue);

  // In all circumstances, re-def the super registers. It's definitely a new
  // value now. This doesn't uniquely identify the composition of subregs, for
  // example, two identical values in subregisters composed in different
  // places would not get equal value numbers.
  for (MCSuperRegIterator SRI(DstRegNum, TRI); SRI.isValid(); ++SRI)
    MTracker->defReg(*SRI, CurBB, CurInst);

  // If we're emulating old LiveDebugValues, just define all the subregisters.
  // DBG_VALUEs of them will expect to be tracked from the DBG_VALUE, not
  // through prior copies.
  if (EmulateOldLDV) {
    for (MCSubRegIndexIterator DRI(DstRegNum, TRI); DRI.isValid(); ++DRI)
      MTracker->defReg(DRI.getSubReg(), CurBB, CurInst);
    return;
  }

  // Otherwise, actually copy subregisters from one location to another.
  // XXX: in addition, any subregisters of DstRegNum that don't line up with
  // the source register should be def'd.
  for (MCSubRegIndexIterator SRI(SrcRegNum, TRI); SRI.isValid(); ++SRI) {
    unsigned SrcSubReg = SRI.getSubReg();
    unsigned SubRegIdx = SRI.getSubRegIndex();
    unsigned DstSubReg = TRI->getSubReg(DstRegNum, SubRegIdx);
    if (!DstSubReg)
      continue;

    // Do copy. There are two matching subregisters, the source value should
    // have been def'd when the super-reg was, the latter might not be tracked
    // yet.
    ValueIDNum CpyValue = SrcValue;

    // This will force SRcSubReg to be tracked, if it isn't yet.
    (void)MTracker->readReg(SrcSubReg);
    LocIdx SrcL = MTracker->getRegMLoc(SrcSubReg);
    assert(SrcL);
    (void)MTracker->readReg(DstSubReg);
    LocIdx DstL = MTracker->getRegMLoc(DstSubReg);
    assert(DstL);
    CpyValue.LocNo = SrcL;

    MTracker->setReg(DstSubReg, CpyValue);
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

Optional<SpillLoc> LiveDebugValues::isRestoreInstruction(const MachineInstr &MI,
                                                         MachineFunction *MF,
                                                         unsigned &Reg) {
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
  // will have changed. DbgEntityHistoryCalculator doesn't try to detect this.
  if (isSpillInstruction(MI, MF)) {
    Loc = extractSpillBaseRegAndOffset(MI);

    if (TTracker) {
      LocIdx MLoc = MTracker->getSpillMLoc(*Loc);
      if (MLoc != 0)
        TTracker->clobberMloc(MLoc, MI.getIterator());
    }
  }

  // Try to recognise spill and restore instructions that may transfer a value.
  if (isLocationSpill(MI, MF, Reg)) {
    Loc = extractSpillBaseRegAndOffset(MI);
    auto ValueID = MTracker->readReg(Reg);

    // If the location is empty, produce a phi, signify it's the live-in value.
    if (ValueID.LocNo == 0)
      ValueID = {CurBB, 0, MTracker->getRegMLoc(Reg)};

    MTracker->setSpill(*Loc, ValueID);
    assert(MTracker->getSpillMLoc(*Loc) != 0);

    // Tell TransferTracker about this spill, produce DBG_VALUEs for it.
    if (TTracker)
      TTracker->transferMlocs(MTracker->getRegMLoc(Reg),
                              MTracker->getSpillMLoc(*Loc), MI.getIterator());

    // Old LiveDebugValues would, at this point, stop tracking the source
    // register of the store.
    if (EmulateOldLDV) {
      for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
        MTracker->defReg(*RAI, CurBB, CurInst);
    }
  } else {
    if (!(Loc = isRestoreInstruction(MI, MF, Reg)))
      return false;

    // Is there a value to be restored?
    auto ValueID = MTracker->readSpill(*Loc);
    if (ValueID.LocNo != 0) {
      // XXX -- can we recover sub-registers of this value? Until we can, first
      // overwrite all defs of the register being restored to.
      for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
        MTracker->defReg(*RAI, CurBB, CurInst);

      // Now override the reg we're restoring to.
      MTracker->setReg(Reg, ValueID);
      assert(MTracker->getSpillMLoc(*Loc) != 0);

      // Report this restore to the transfer tracker too.
      if (TTracker)
        TTracker->transferMlocs(MTracker->getSpillMLoc(*Loc),
                                MTracker->getRegMLoc(Reg), MI.getIterator());
    } else {
      // There isn't anything in the location; not clear if this is a code path
      // that still runs. Def this register anyway just in case.
      for (MCRegAliasIterator RAI(Reg, TRI, true); RAI.isValid(); ++RAI)
        MTracker->defReg(*RAI, CurBB, CurInst);

      // Set the restored value to be a machine phi number, signifying that it's
      // whatever the spills live-in value is in this block.
      LocIdx L = MTracker->getSpillMLoc(*Loc);
      ValueID = {CurBB, 0, L};
      MTracker->setReg(Reg, ValueID);
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
  // soon. It is more likely that previous register, which is callee saved, is
  // going to stay unclobbered longer, even if it is killed.
  //
  // For new LiveDebugValues, we can track multiple locations per value, so
  // ignore this condition.
  if (EmulateOldLDV && !isCalleeSavedReg(DestReg))
    return false;

  // Old LiveDebugValues only followed killing copies.
  if (EmulateOldLDV && !SrcRegOp->isKill())
    return false;

  // We have to follow identity copies, as DbgEntityHistoryCalculator only
  // sees the defs. XXX is this code path still taken?
  //auto ValueID = MTracker->readReg(SrcReg);
  //MTracker->setReg(DestReg, ValueID);
  // Copy MTracker info, including subregs if available.
  LiveDebugValues::performCopy(SrcReg, DestReg);

  // Only produce a transfer of DBG_VALUE within a block where old LDV
  // would have. We might make use of the additional value tracking in some
  // other way, later.
  if (TTracker && isCalleeSavedReg(DestReg) && SrcRegOp->isKill())
    TTracker->transferMlocs(MTracker->getRegMLoc(SrcReg),
                            MTracker->getRegMLoc(DestReg), MI.getIterator());

  // Old LiveDebugValues would quit tracking the old location after copying.
  if (EmulateOldLDV && SrcReg != DestReg)
    MTracker->defReg(SrcReg, CurBB,  CurInst);

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

void LiveDebugValues::process(MachineInstr &MI, uint64_t **MInLocs) {
  // Try to interpret an MI as a debug or transfer instruction. Only if it's
  // none of these should we interpret it's register defs as new value
  // definitions.
  if (transferDebugValue(MI))
    return;
  if (transferDebugInstrRef(MI, MInLocs))
    return;
  if (transferRegisterCopy(MI))
    return;
  if (transferSpillOrRestoreInst(MI))
    return;
  transferRegisterDef(MI);
}

void LiveDebugValues::produceMLocTransferFunction(
    MachineFunction &MF, SmallVectorImpl<MLocTransferMap> &MLocTransfer,
    unsigned MaxNumBlocks) {
  // Because we try to optimize around register mask operands by ignoring regs
  // that aren't currently tracked, we set up something ugly for later: RegMask
  // operands that are seen earlier than the first use of a register, still need
  // to clobber that register in the transfer function. But this information
  // isn't actively recorded. Instead, we track each RegMask used in each block,
  // and accumulated the clobbered but untracked registers in each block into
  // the following bitvector. Later, if new values are tracked, we can add
  // appropriate clobbers.
  SmallVector<BitVector, 32> BlockMasks;
  BlockMasks.resize(MaxNumBlocks);

  // Reserve one bit per register for the masks described above.
  unsigned BVWords = MachineOperand::getRegMaskSize(TRI->getNumRegs());
  for (auto &BV : BlockMasks)
    BV.resize(TRI->getNumRegs(), true);

  // Step through all instructions and inhale the transfer function.
  for (auto &MBB : MF) {
    // Object fields that are read by trackers to know where we are in the
    // function.
    CurBB = MBB.getNumber();
    CurInst = 1;

    // Set all machine locations to a PHI value. For transfer function
    // production only, this signifies the live-in value to the block.
    MTracker->reset();
    MTracker->setMPhis(CurBB);

    // Step through each instruction in this block.
    for (auto &MI : MBB) {
      process(MI);
      // Also accumulate fragment map.
      if (MI.isDebugValue())
        accumulateFragmentMap(MI);

      if (MI.peekDebugValueID() != 0) {
        uint64_t instid = MI.peekDebugValueID();
        assert(InstrIDMap.find(instid) == InstrIDMap.end());
        InstrIDMap[instid] = std::make_pair(&MI, CurInst);

        // Also read all known values at this point.
        // Expensive copy, woo!
        DebugReadPoints[instid] = std::make_pair(&MI, MTracker->LocIdxToIDNum);
      }

      ++CurInst;
    }

    // Produce the transfer function, a map of machine location to new value. If
    // any machine location has the live-in phi value from the start of the
    // block, it's live-through and doesn't need recording in the transfer
    // function.
    for (unsigned IdxNum = 1; IdxNum < MTracker->getNumLocs(); ++IdxNum) {
      LocIdx Idx = LocIdx(IdxNum);
      ValueIDNum P = MTracker->getNumAtPos(Idx);
      if (P.InstNo == 0 && P.LocNo == Idx)
        continue;

      MLocTransfer[CurBB][Idx] = P;
    }

    // Accumulate any bitmask operands into the clobberred reg mask for this
    // block.
    for (auto &P : MTracker->Masks) {
      BlockMasks[CurBB].clearBitsNotInMask(P.first->getRegMask(), BVWords);
    }
  }

  // Compute a bitvector of all the registers that are tracked in this block.
  const TargetLowering *TLI = MF.getSubtarget().getTargetLowering();
  unsigned SP = TLI->getStackPointerRegisterToSaveRestore();
  BitVector UsedRegs(TRI->getNumRegs());
  for (auto &P : MTracker->LocIdxToLocID) {
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
      unsigned ID = MTracker->getLocID(Bit, false);
      LocIdx Idx = MTracker->LocIDToLocIdx[ID];
      assert(Idx != 0);
      ValueIDNum &ValueID = MLocTransfer[I][Idx];
      if (ValueID.BlockNo == I && ValueID.InstNo == 0)
        // it was left as live-through. Set it to clobbered.
        ValueID = ValueIDNum{0, 0, LocIdx(0)};
    }
  }
}

bool LiveDebugValues::mlocJoin(
    MachineBasicBlock &MBB, SmallPtrSet<const MachineBasicBlock *, 16> &Visited,
    uint64_t **OutLocs, uint64_t *InLocs) {
  LLVM_DEBUG(dbgs() << "join MBB: " << MBB.getNumber() << "\n");
  bool Changed = false;

  // Collect predecessors that have been visited. Anything that hasn't been
  // visited yet is a backedge on the first iteration, and the meet of it's
  // lattice value for all locations will be unaffected.
  SmallVector<const MachineBasicBlock *, 8> BlockOrders;
  for (auto Pred : MBB.predecessors()) {
    if (Visited.count(Pred)) {
      BlockOrders.push_back(Pred);
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

  // Step through all machine locations, then look at each predecessor and
  // detect disagreements.
  unsigned ThisBlockRPO = BBToOrder.find(&MBB)->second;
  for (unsigned Idx = 1; Idx < MTracker->getNumLocs(); ++Idx) {
    // Pick out the first predecessors live-out value for this location. It's
    // guaranteed to be not a backedge, as we order by RPO.
    uint64_t BaseVal = OutLocs[BlockOrders[0]->getNumber()][Idx];

    // Some flags for whether there's a disagreement, and whether it's a
    // disagreement with a backedge or not.
    bool Disagree = false;
    bool NonBackEdgeDisagree = false;

    // Loop around everything that wasn't 'base'.
    for (unsigned int I = 1; I < BlockOrders.size(); ++I) {
      auto *MBB = BlockOrders[I];
      if (BaseVal != OutLocs[MBB->getNumber()][Idx]) {
        // Live-out of a predecessor disagrees with the first predecessor.
        Disagree = true;

        // Test whether it's a disagreemnt in the backedges or not.
        if (BBToOrder.find(MBB)->second < ThisBlockRPO) // might be self b/e
          NonBackEdgeDisagree = true;
      }
    }

    bool OverRide = false;
    if (Disagree && !NonBackEdgeDisagree &&
        ValueIDNum::fromU64(InLocs[Idx]).LocNo != 0) {
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
      ValueIDNum BaseID = ValueIDNum::fromU64(BaseVal);
      unsigned BaseBlockRPONum = BBNumToRPO[BaseID.BlockNo] + 1;
      if (BaseID.InstNo != 0)
        BaseBlockRPONum = 0;

      ValueIDNum InLocID = ValueIDNum::fromU64(InLocs[Idx]);
      unsigned InLocRPONum = BBNumToRPO[InLocID.BlockNo] + 1;
      if (InLocID.InstNo != 0)
        InLocRPONum = 0;

      // Should we ignore the disagreeing backedges, and override with the
      // value the other predecessors agree on (in "base")?
      unsigned ThisBlockRPONum = BBNumToRPO[MBB.getNumber()] + 1;
      if (BaseBlockRPONum > InLocRPONum && BaseBlockRPONum < ThisBlockRPONum) {
        // Override.
        OverRide = true;
      }
    }
    // else: if we disagree in the non-backedges, then this is definitely
    // a control flow merge where different values merge. Make it a PHI.

    // Generate a phi...
    ValueIDNum PHI = {(uint64_t)MBB.getNumber(), 0, LocIdx(Idx)};
    uint64_t NewVal = (Disagree && !OverRide) ? PHI.asU64() : BaseVal;
    if (InLocs[Idx] != NewVal) {
      Changed |= true;
      InLocs[Idx] = NewVal;
    }
  }

  // Uhhhhhh, reimplement NumInserted and NumRemoved pls.
  return Changed;
}

void LiveDebugValues::mlocDataflow(uint64_t **MInLocs, uint64_t **MOutLocs,
                                   SmallVectorImpl<MLocTransferMap> &MLocTransfer) {
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Worklist, Pending;

  for (unsigned int I = 0; I < BBToOrder.size(); ++I)
    Worklist.push(I);

  MTracker->reset();

  // Set inlocs for entry block -- each as a PHI at the entry block. Represents
  // the incoming value to the function.
  MTracker->setMPhis(0);
  for (unsigned Idx = 1; Idx < MTracker->getNumLocs(); ++Idx) {
    ValueIDNum Val = MTracker->getNumAtPos(LocIdx(Idx));
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
    SmallVector<std::pair<LocIdx, ValueIDNum>, 32> ToRemap;

    while (!Worklist.empty()) {
      MachineBasicBlock *MBB = OrderToBB[Worklist.top()];
      CurBB = MBB->getNumber();
      Worklist.pop();

      // Join the values in all predecessor blocks.
      bool InLocsChanged = mlocJoin(*MBB, Visited, MOutLocs, MInLocs[CurBB]);
      InLocsChanged |= Visited.insert(MBB).second;

      // Don't examine transfer function if we've visited this loc at least
      // once, and inlocs haven't changed.
      if (!InLocsChanged)
        continue;

      // Load the current set of live-ins into MLocTracker.
      MTracker->loadFromArray(MInLocs[CurBB], CurBB);

      // Each element of the transfer function can be a new def, or a read of
      // a live-in value. Evaluate each element, and store to "ToRemap".
      ToRemap.clear();
      for (auto &P : MLocTransfer[CurBB]) {
        ValueIDNum NewID = {0, 0, LocIdx(0)};
        if (P.second.BlockNo == CurBB && P.second.InstNo == 0) {
          // This is a movement of whatever was live in. Read it.
          NewID = MTracker->getNumAtPos(P.second.LocNo);
        } else {
          // It's a def. Just set it.
          assert(P.second.BlockNo == CurBB || P.second.LocNo == 0);
          NewID = P.second;
        }
        ToRemap.push_back(std::make_pair(P.first, NewID));
      }

      // Commit the transfer function changes into mloc tracker, which
      // transforms the contents of the MLocTracker into the live-outs.
      for (auto &P : ToRemap)
        MTracker->setMLoc(P.first, P.second);

      // Now copy out-locs from mloc tracker into out-loc vector, checking
      // whether changes have occurred. These changes can have come from both
      // the transfer function, and mlocJoin.
      bool OLChanged = false;
      for (unsigned Idx = 1; Idx < MTracker->getNumLocs(); ++Idx) {
        uint64_t ID = MTracker->getNumAtPos(LocIdx(Idx)).asU64();
        OLChanged |= MOutLocs[CurBB][Idx] != ID;
        MOutLocs[CurBB][Idx] = ID;
      }

      MTracker->reset();

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

  // Once all the live-ins don't change on mlocJoin(), we've reached a
  // fixedpoint.
}

bool LiveDebugValues::vlocJoinLocation(
    MachineBasicBlock &MBB, const ValueRec &InLoc, const ValueRec &OLoc,
    uint64_t *InLocOutLocs, uint64_t *OLOutLocs,
    const LiveIdxT::mapped_type PrevInLocs, // ptr
    const DebugVariable &CurVar, bool ThisIsABackEdge) {
  // This method checks whether InLoc and OLoc, the values of a variable
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

  unsigned BBNum = MBB.getNumber();
  bool EarlyBail = false;

  // Lambda to pick a machine location for a value, if we decide to use a PHI
  // and merge them. This could be much more sophisticated, but right now
  // is good enough. When emulating old LiveDebugValues, there should only be
  // one candidate location for a value anyway.
  auto FindLocInLocs = [&](uint64_t *OutLocs, const ValueIDNum &ID) -> LocIdx {
    unsigned NumLocs = MTracker->getNumLocs();
    LocIdx theloc = LocIdx(0);
    for (unsigned i = 0; i < NumLocs; ++i) {
      if (OutLocs[i] == ID.asU64()) {
        if (theloc != 0) {
          // Prefer non-spills
          if (MTracker->isSpill(theloc))
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
  EarlyBail |= (InLoc.Kind == OLoc.Kind && InLoc.Kind == ValueRec::Const &&
                !InLoc.MO->isIdenticalTo(*OLoc.MO));

  // Meta disagreement -> bail early. We wouldn't be able to produce a
  // DBG_VALUE that reconciled the Meta information.
  EarlyBail |= (InLoc.meta != OLoc.meta);

  // LocNo == 0 (undef) -> bail early.
  EarlyBail |= (InLoc.Kind == OLoc.Kind && InLoc.Kind == ValueRec::Def &&
                OLoc.ID.LocNo == 0);

  // Bail out if early bail signalled.
  if (EarlyBail) {
    return false;
  } else if (InLoc.Kind == ValueRec::Const) {
    // If both are constants and we didn't early-bail, they're the same.
    return true;
  }

  // This is a join for "values". Two important facts: is this a backedge, and
  // does InLocs refer to a machine location PHI already?
  assert(InLoc.Kind == ValueRec::Def);
  const ValueIDNum &InLocsID = InLoc.ID;
  const ValueIDNum &OLID = OLoc.ID;
  bool ThisIsAnMPHI = InLocsID.BlockNo == BBNum && InLocsID.InstNo == 0;

  // Find a machine location for the OLID in its out-locs.
  LocIdx OLIdx = FindInOLocs(OLID);

  // Everything is massively different for backedges. Try not-be's first.
  if (!ThisIsABackEdge) {
    // If both values agree, no more work is required, and a location can be
    // picked for the value later, when DBG_VALUEs are created.
    // However if they disagree, or the value is a PHI in this block, then
    // we may need to create a new PHI, or verify that the correct values flow
    // into the machine location PHI.
    if (InLoc == OLoc && !ThisIsAnMPHI)
      return true;

    // If we're non-identical and there's no mphi, definitely can't merge.
    // XXX document that InLoc is always the mphi, if ther eis noe.
    if (InLoc != OLoc && !ThisIsAnMPHI)
      return false;

    // Otherwise, we're definitely an mphi, and need to prove that the
    // value from OLoc feeds into it. Because we're an mphi, we know
    // our location:
    LocIdx InLocIdx = InLocsID.LocNo;
    // That value must be live-out of the predecessor, in the location for
    // that mphi.
    bool HasMOutLoc = OLOutLocs[InLocIdx] == OLID.asU64();
    if (!HasMOutLoc)
      // They conflict and/or are in the wrong location. Incompatible.
      return false;
    return true;
  }

  // If the backedge value has no location, definitely can't merge.
  if (OLIdx == 0)
    return false;

  LocIdx Idx = FindInInLocs(InLocsID);
  if (Idx == 0 && InLocsID.BlockNo == BBNum && InLocsID.InstNo == 0)
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
    // fatal to this variable.
    return false;

  ValueIDNum &ILS_ID = ILS_It->second.ID;
  unsigned NewInOrder =
      (InLocsID.InstNo) ? 0 : BBNumToRPO[InLocsID.BlockNo] + 1;
  unsigned OldOrder = (ILS_ID.InstNo) ? 0 : BBNumToRPO[ILS_ID.BlockNo] + 1;
  if (OldOrder >= NewInOrder)
    return false;

  return true;
}

bool LiveDebugValues::vlocJoin(
    MachineBasicBlock &MBB, LiveIdxT &VLOCOutLocs, LiveIdxT &VLOCInLocs,
    SmallPtrSet<const MachineBasicBlock *, 16> *VLOCVisited, unsigned BBNum,
    const SmallSet<DebugVariable, 4> &AllVars, uint64_t **MInLocs,
    uint64_t **MOutLocs,
    SmallPtrSet<const MachineBasicBlock *, 8> &NonAssignBlocks) {

  // To emulate old LiveDebugValues, process this block if it's not in scope but
  // _does_ assign a variable value. No live-ins for this scope are transferred
  // in though, so we can return immediately.
  if (NonAssignBlocks.count(&MBB) == 0 && !ArtificialBlocks.count(&MBB)) {
    if (VLOCVisited)
      return true;
    return false;
  }

  LLVM_DEBUG(dbgs() << "join MBB: " << MBB.getNumber() << "\n");
  bool Changed = false;

  // Map that we'll be using to store the computed live-ins.
  DenseMap<DebugVariable, ValueRec> InLocsT;

  // Find any live-ins computed in a prior iteration.
  auto ILSIt = VLOCInLocs.find(&MBB);
  assert(ILSIt != VLOCInLocs.end());
  auto &ILS = *ILSIt->second;

  // Helper to pick a live-out location for a value. Much like in mlocJoin.
  // Could be much more sophisticated, but doesn't need to be while we're
  // emulating old LiveDebugValues.
  auto FindLocOfDef = [&](unsigned BBNum, const ValueIDNum &ID) -> LocIdx {
    unsigned NumLocs = MTracker->getNumLocs();
    uint64_t *OutLocs = MOutLocs[BBNum];
    LocIdx theloc = LocIdx(0);
    for (unsigned i = 0; i < NumLocs; ++i) {
      if (OutLocs[i] == ID.asU64()) {
        if (theloc != 0) {
          // Prefer non-spills
          if (MTracker->isSpill(theloc))
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

  // Order predecessors by RPOT order, for exploring them in that order.
  SmallVector<MachineBasicBlock *, 8> BlockOrders;
  for (auto p : MBB.predecessors())
    BlockOrders.push_back(p);

  auto Cmp = [&](MachineBasicBlock *A, MachineBasicBlock *B) {
    return BBToOrder[A] < BBToOrder[B];
  };

  llvm::sort(BlockOrders.begin(), BlockOrders.end(), Cmp);

  unsigned ThisBlockRPONum = BBToOrder[&MBB];

  // For all predecessors of this MBB, find the set of variable values that
  // can be joined.
  int NumVisited = 0;
  unsigned FirstVisited = 0;
  for (auto p : BlockOrders) {
    // Ignore backedges if we have not visited the predecessor yet. As the
    // predecessor hasn't yet had values propagated into it, all variables will
    // have an "Unknown" lattice value that meets with everything.
    // If a value guessed to be correct here is invalidated later, we will
    // remove it when we revisit this block.
    if (VLOCVisited && !VLOCVisited->count(p)) {
      LLVM_DEBUG(dbgs() << "  ignoring unvisited pred MBB: " << p->getNumber()
                        << "\n");
      continue;
    }

    auto OL = VLOCOutLocs.find(p);
    // Join is null if any predecessors OutLocs is absent or empty.
    if (OL == VLOCOutLocs.end()) {
      InLocsT.clear();
      break;
    }

    // For the first predecessor, copy all of its variable values into the
    // InLocsT map; check whether other predecessors join with each value
    // later.
    if (!NumVisited) {
      InLocsT = *OL->second;
      FirstVisited = p->getNumber();

      // Additionally: for each variable, check whether the CFG join carries
      // the value into this block unchanged, or whether an mphi happens,
      // according to the machine value analysis. If it isn't an mphi, the
      // result of joining this location must be that value (or fail). If it is,
      // it has to be that mphi value (or fail).
      // XXX, this might be a better way to decompose the joining problem?

      for (auto &It : InLocsT) {
        // Consider only defs,
        if (It.second.Kind != ValueRec::Def)
          continue;
        // Does it have a live-out machine location?
        LocIdx Idx = FindLocOfDef(FirstVisited, It.second.ID);
        if (Idx == 0)
          continue;
        // And is that what's in the corresponding live-in machine location?
        ValueIDNum LiveInID = ValueIDNum::fromU64(MInLocs[BBNum][Idx]);
        if (It.second.ID != LiveInID) {
          // No, it became an mphi. Turn the candidate live-in location to that
          // mphi, and check the other predecessors later.
          assert(LiveInID.BlockNo == BBNum && LiveInID.InstNo == 0);
          It.second.ID = LiveInID;
        }
      }
    } else {
      // Check whether this predecessors variable values will successfully
      // join with the first predecessors value, or mphi.
      for (auto &Var : AllVars) {
        auto InLocsIt = InLocsT.find(Var);
        auto OLIt = OL->second->find(Var);

        // Regardless of what's being joined in, an empty predecessor means
        // there can be no incoming value here.
        if (InLocsIt == InLocsT.end())
          continue;

        if (OLIt == OL->second->end()) {
          InLocsT.erase(InLocsIt);
          continue;
        }

        bool ThisIsABackEdge = ThisBlockRPONum <= BBToOrder[p];
        bool joins = vlocJoinLocation(
            MBB, InLocsIt->second, OLIt->second, MOutLocs[FirstVisited],
            MOutLocs[p->getNumber()], &ILS, InLocsIt->first, ThisIsABackEdge);

        // If we cannot join the two values, erase the live-in variable.
        if (!joins)
          InLocsT.erase(InLocsIt);
      }
    }

    // xXX jmorse deleted debug statement

    NumVisited++;
  }

  // Store newly calculated in-locs into VLOCInLocs, if they've changed.
  Changed = ILS != InLocsT;
  if (Changed)
    ILS = std::move(InLocsT);

  // Uhhhhhh, reimplement NumInserted and NumRemoved pls.
  return Changed;
}

void LiveDebugValues::vlocDataflow(
    const LexicalScope *Scope, const DILocation *DILoc,
    const SmallSet<DebugVariable, 4> &VarsWeCareAbout,
    SmallPtrSetImpl<MachineBasicBlock *> &AssignBlocks, LiveInsT &Output,
    uint64_t **MOutLocs, uint64_t **MInLocs,
    SmallVectorImpl<VLocTracker> &AllTheVLocs) {
  // This method is much like mlocDataflow: but focuses on a single
  // LexicalScope at a time. Pick out a set of blocks and variables that are
  // to have their value assignments solved, then run our dataflow algorithm
  // until a fixedpoint is reached.
  std::priority_queue<unsigned int, std::vector<unsigned int>,
                      std::greater<unsigned int>>
      Worklist, Pending;

  // The set of blocks we'll be examining.
  SmallPtrSet<const MachineBasicBlock *, 8> LBlocks;
  // The order in which to examine them (RPO).
  SmallVector<MachineBasicBlock *, 8> BlockOrders;

  // RPO ordering function.
  auto Cmp = [&](MachineBasicBlock *A, MachineBasicBlock *B) {
    return BBToOrder[A] < BBToOrder[B];
  };

  LS.getMachineBasicBlocks(DILoc, LBlocks);

  // A separate container to distinguish "blocks we're exploring" versus
  // "blocks that are potentially in scope. See comment at start of vlocJoin.
  SmallPtrSet<const MachineBasicBlock *, 8> NonAssignBlocks = LBlocks;

  // Old LiveDebugValues tracks variable locations that come out of blocks
  // not in scope, where DBG_VALUEs occur. This is something we could
  // legitimately ignore, but lets allow it for now.
  if (EmulateOldLDV)
    LBlocks.insert(AssignBlocks.begin(), AssignBlocks.end());


  // Accumulate in any artificial blocks that immediately follow any of those
  // blocks.
  DenseSet<const MachineBasicBlock *> ToAdd;
  auto AccumulateArtificialBlocks = [this, &ToAdd, &LBlocks, &NonAssignBlocks](const MachineBasicBlock* MBB) {
    SmallVector<std::pair<const MachineBasicBlock *, MachineBasicBlock::const_succ_iterator>, 8> DFS;
    // Find any artificial successors not already tracked.
    for (auto *succ : MBB->successors()) {
      if (LBlocks.count(succ) || NonAssignBlocks.count(succ))
        continue;
      if (!ArtificialBlocks.count(succ))
        continue;
      DFS.push_back(std::make_pair(succ, succ->succ_begin()));
      ToAdd.insert(succ);
    }

    // Search all those blocks, depth first.
    while (!DFS.empty()) {
      const MachineBasicBlock *CurBB = DFS.back().first;
      MachineBasicBlock::const_succ_iterator &CurSucc = DFS.back().second;
      if (CurSucc == CurBB->succ_end()) {
        DFS.pop_back();
        continue;
      }

      if (!ToAdd.count(*CurSucc) && ArtificialBlocks.count(*CurSucc)) {
        DFS.push_back(std::make_pair(*CurSucc, (*CurSucc)->succ_begin()));
        ToAdd.insert(*CurSucc);
        continue;
      }

      ++CurSucc;
    }
  };

  for (auto *MBB : LBlocks)
    AccumulateArtificialBlocks(MBB);
  for (auto *MBB : NonAssignBlocks)
    AccumulateArtificialBlocks(MBB);
    
  LBlocks.insert(ToAdd.begin(), ToAdd.end());
  NonAssignBlocks.insert(ToAdd.begin(), ToAdd.end());

  // Single block scope: not interesting! No propagation at all. Note that
  // this could probably go above ArtificialBlocks without damage, but
  // that then produces output differences from original-live-debug-values,
  // which propagates from a single block into many artificial ones.
  if (LBlocks.size() == 1)
    return;

  // Picks out relevants blocks RPO order and sort them.
  for (auto *MBB : LBlocks)
    BlockOrders.push_back(const_cast<MachineBasicBlock *>(MBB));

  llvm::sort(BlockOrders.begin(), BlockOrders.end(), Cmp);
  unsigned NumBlocks = BlockOrders.size();

  // Allocate some vectors for storing the live ins and live outs. Large.
  SmallVector<DenseMap<DebugVariable, ValueRec>, 32> LiveIns, LiveOuts;
  LiveIns.resize(NumBlocks);
  LiveOuts.resize(NumBlocks);

  // Produce by-MBB indexes of live-in/live-outs, to ease lookup within
  // vlocJoin.
  LiveIdxT LiveOutIdx, LiveInIdx;
  LiveOutIdx.reserve(NumBlocks);
  LiveInIdx.reserve(NumBlocks);
  for (unsigned I = 0; I < NumBlocks; ++I) {
    LiveOutIdx[BlockOrders[I]] = &LiveOuts[I];
    LiveInIdx[BlockOrders[I]] = &LiveIns[I];
  }

  for (auto *MBB : BlockOrders)
    Worklist.push(BBToOrder[MBB]);

  bool FirstTrip = true;
  SmallPtrSet<const MachineBasicBlock *, 16> VLOCVisited;
  while (!Worklist.empty() || !Pending.empty()) {
    SmallPtrSet<MachineBasicBlock *, 16> OnPending;
    while (!Worklist.empty()) {
      auto *MBB = OrderToBB[Worklist.top()];
      CurBB = MBB->getNumber();
      Worklist.pop();

      // Join values from predecessors.
      bool InlocsChanged = vlocJoin(
          *MBB, LiveOutIdx, LiveInIdx, (FirstTrip) ? &VLOCVisited : nullptr,
          CurBB, VarsWeCareAbout, MInLocs, MOutLocs, NonAssignBlocks);

      // Always explore transfer function if inlocs changed, or if we've not
      // visited this block before.
      InlocsChanged |= VLOCVisited.insert(MBB).second;
      if (!InlocsChanged)
        continue;

      // Do transfer function.
      // DenseMap copy.
      DenseMap<DebugVariable, ValueRec> Cpy = *LiveInIdx[MBB];
      auto &VTracker = AllTheVLocs[MBB->getNumber()];
      for (auto &Transfer : VTracker.Vars) {
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
    FirstTrip = false;
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

void LiveDebugValues::dump_mloc_transfer(
    const MLocTransferMap &mloc_transfer) const {
  for (auto &P : mloc_transfer) {
    std::string foo = MTracker->LocIdxToName(P.first);
    std::string bar = MTracker->IDAsString(P.second);
    dbgs() << "Loc " << foo << " --> " << bar << "\n";
  }
}

void LiveDebugValues::emitLocations(
    MachineFunction &MF, LiveInsT SavedLiveIns, uint64_t **MInLocs,
    DenseMap<DebugVariable, unsigned> &AllVarsNumbering) {
  TTracker = new TransferTracker(TII, MTracker, MF, *TRI, CalleeSavedRegs);
  unsigned NumLocs = MTracker->getNumLocs();

  // For each block, load in the machine value locations and variable value
  // live-ins, then step through each instruction in the block. New DBG_VALUEs
  // to be inserted will be created along the way.
  for (MachineBasicBlock &MBB : MF) {
    unsigned bbnum = MBB.getNumber();
    MTracker->reset();
    MTracker->loadFromArray(MInLocs[bbnum], bbnum);
    TTracker->loadInlocs(MBB, MInLocs[bbnum], SavedLiveIns[MBB.getNumber()],
                         NumLocs, bbnum);

    CurBB = bbnum;
    CurInst = 1;
    for (auto &MI : MBB) {
      process(MI, MInLocs);
      TTracker->prodAfterInst(CurInst, MI.getIterator());
      ++CurInst;
    }
  }

  // We have to insert DBG_VALUEs in a consistent order, otherwise they appeaer
  // in DWARF in different orders. Use the order that they appear when walking
  // through each block / each instruction, stored in AllVarsNumbering.
  auto OrderDbgValues = [&](const MachineInstr *A,
                            const MachineInstr *B) -> bool {
    DebugVariable VarA(A->getDebugVariable(), A->getDebugExpression(),
                       A->getDebugLoc()->getInlinedAt());
    DebugVariable VarB(B->getDebugVariable(), B->getDebugExpression(),
                       B->getDebugLoc()->getInlinedAt());
    return AllVarsNumbering.find(VarA)->second <
           AllVarsNumbering.find(VarB)->second;
  };

  // Go through all the transfers recorded in the TransferTracker -- this is
  // both the live-ins to a block, and any movements of values that happen
  // in the middle.
  for (auto &P : TTracker->Transfers) {
    // Sort them according to appearance order.
    llvm::sort(P.Insts.begin(), P.Insts.end(), OrderDbgValues);
    // Insert either before or after the designated point...
    if (P.MBB) {
      MachineBasicBlock &MBB = *P.MBB;
      for (auto *MI : P.Insts) {
        MBB.insert(P.Pos, MI);
      }
    } else {
      MachineBasicBlock &MBB = *P.Pos->getParent();
      for (auto *MI : P.Insts) {
        MBB.insertAfter(P.Pos, MI);
      }
    }
  }
}

void LiveDebugValues::initialSetup(MachineFunction &MF) {
  // Build some useful data structures.
  auto hasNonArtificialLocation = [](const MachineInstr &MI) -> bool {
    if (const DebugLoc &DL = MI.getDebugLoc())
      return DL.getLine() != 0;
    return false;
  };
  // Collect a set of all the artificial blocks.
  for (auto &MBB : MF)
    if (none_of(MBB.instrs(), hasNonArtificialLocation))
      ArtificialBlocks.insert(&MBB);

  // Compute mappings of block <=> RPO order.
  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);
  unsigned int RPONumber = 0;
  for (auto RI = RPOT.begin(), RE = RPOT.end(); RI != RE; ++RI) {
    OrderToBB[RPONumber] = *RI;
    BBToOrder[*RI] = RPONumber;
    BBNumToRPO[(*RI)->getNumber()] = RPONumber;
    ++RPONumber;
  }
}

namespace {
class jmorseblock;

class jmorsephi {
public:
  SmallVector<std::pair<jmorseblock*, uint64_t>, 4> vec;
  jmorseblock *parent;
  uint64_t theval = 0;
  jmorsephi(uint64_t theval, jmorseblock *parent) : parent(parent), theval(theval) { }

  jmorseblock *getParent() {
    return parent;
  }
};

class jmorseupdater;
class jmorseblock;

class jmorseblockit {
public:
  MachineBasicBlock::pred_iterator pit;
  jmorseupdater &j;

  jmorseblockit(MachineBasicBlock::pred_iterator pit, jmorseupdater &j)
    : pit(pit), j(j) { }

  bool operator!=(const jmorseblockit &jit) const {
    return jit.pit != pit;
  }

  jmorseblockit &operator++() {
    ++pit;
    return *this;
  }

  jmorseblock *operator*();
};

class jmorseblock {
public:
  MachineBasicBlock &BB;
  jmorseupdater &j;
  jmorseblock(MachineBasicBlock &BB, jmorseupdater &j) : BB(BB), j(j) {
  }

  jmorseblockit succ_begin() {
    return jmorseblockit(BB.succ_begin(), j);
  }

  jmorseblockit succ_end() {
    return jmorseblockit(BB.succ_end(), j);
  }

  // Good news! there are no phis.
  // XXX -- this assumes that we don't need to look up created phis via this
  // method, which would suck.
  using philist = std::vector<jmorsephi>;
  iterator_range<typename philist::iterator> phis() {
    return make_range(nullptr, nullptr);
  }
};

class jmorseupdater {
public:
  std::map<uint64_t, jmorsephi *> phis;
  std::map<MachineBasicBlock *, SmallSet<uint64_t, 4>> phi_map;
  std::set<uint64_t> lolundefs;
  std::map<MachineBasicBlock *, uint64_t> undef_map;
  // This is set high and decremented, on the assumption it isn't going
  // to alias with valueid numbers.
  uint64_t value_count = std::numeric_limits<uint64_t>::max();


  std::map<MachineBasicBlock *, jmorseblock *> bb_map;

  void reset() {
    for (auto &p : phis) {
      delete p.second;
    }
    phis.clear();
    lolundefs.clear();
    value_count = std::numeric_limits<uint64_t>::max();
  }

  ~jmorseupdater() {
    reset();
  }

  uint64_t get_new_val() {
    return value_count--;
  }

  jmorseblock *getjmorsebb(MachineBasicBlock *BB) {
    auto it = bb_map.find(BB);
    if (it == bb_map.end()) {
      bb_map[BB] = new jmorseblock(*BB, *this);
      it = bb_map.find(BB);
    }
    return it->second;
  }
};

jmorseblock *jmorseblockit::operator*() {
  return j.getjmorsebb(*pit);
}

} // anon ns

namespace llvm {

raw_ostream &operator<<(raw_ostream &out, const jmorsephi &phi) {
  out << "jmorsehpi " << phi.theval;
  return out;
}

template<>
class SSAUpdaterTraits<jmorseupdater> {
public:
  using BlkT = jmorseblock;
  using ValT = uint64_t;
  using PhiT = jmorsephi;
  using BlkSucc_iterator = jmorseblockit;

  static BlkSucc_iterator BlkSucc_begin(BlkT *BB) { return BB->succ_begin(); }
  static BlkSucc_iterator BlkSucc_end(BlkT *BB) { return BB->succ_end(); }

  /// Iterator for PHI operands.
  class PHI_iterator {
  private:
    jmorsephi *PHI;
    unsigned idx;

  public:
    explicit PHI_iterator(jmorsephi *P) // begin iterator
      : PHI(P), idx(0) {}
    PHI_iterator(jmorsephi *P, bool) // end iterator
      : PHI(P), idx(PHI->vec.size()) {}

    PHI_iterator &operator++() { idx++; return *this; }
    bool operator==(const PHI_iterator& x) const { return idx == x.idx; }
    bool operator!=(const PHI_iterator& x) const { return !operator==(x); }

    uint64_t getIncomingValue() { return PHI->vec[idx].second; }

    jmorseblock *getIncomingBlock() {
      return PHI->vec[idx].first;
    }
  };

  static inline PHI_iterator PHI_begin(PhiT *PHI) { return PHI_iterator(PHI); }

  static inline PHI_iterator PHI_end(PhiT *PHI) {
    return PHI_iterator(PHI, true);
  }

  /// FindPredecessorBlocks - Put the predecessors of BB into the Preds
  /// vector.
  static void FindPredecessorBlocks(jmorseblock *BB,
                                    SmallVectorImpl<jmorseblock*> *Preds){
    for (MachineBasicBlock::pred_iterator PI = BB->BB.pred_begin(),
           E = BB->BB.pred_end(); PI != E; ++PI)
      Preds->push_back(BB->j.getjmorsebb(*PI));
  }

  /// GetUndefVal - Create an IMPLICIT_DEF instruction with a new register.
  /// Add it into the specified block and return the register.
  static uint64_t GetUndefVal(jmorseblock *BB,
                              jmorseupdater *Updater) {
    // XXX XXX XXX need to record the existance of "something that isn't going
    // to work" somewhere.
    uint64_t n = Updater->get_new_val();
    Updater->lolundefs.insert(n);
    Updater->undef_map[&BB->BB] = n;
    return n;
  }

  /// CreateEmptyPHI - Create a PHI instruction that defines a new register.
  /// Add it into the specified block and return the register.
  static uint64_t CreateEmptyPHI(jmorseblock *BB, unsigned NumPreds,
                                 jmorseupdater *Updater) {
    uint64_t n = Updater->get_new_val();
    jmorsephi *PHI = new jmorsephi(n, BB);
    PHI->theval = n;
    Updater->phis[n] = PHI;
    Updater->phi_map[&BB->BB].insert(n);
    return n;
  }

  /// AddPHIOperand - Add the specified value as an operand of the PHI for
  /// the specified predecessor block.
  static void AddPHIOperand(jmorsephi *PHI, uint64_t Val,
                            jmorseblock *Pred) {
    PHI->vec.push_back(std::make_pair(Pred, Val));
  }

  /// ValueIsPHI - Check if the instruction that defines the specified register
  /// is a PHI instruction.
  static jmorsephi *ValueIsPHI(uint64_t Val, jmorseupdater *Updater) {
    auto it = Updater->phis.find(Val);
    if (it == Updater->phis.end())
      return nullptr;
    return it->second;;
  }

  /// ValueIsNewPHI - Like ValueIsPHI but also check if the PHI has no source
  /// operands, i.e., it was just added.
  static jmorsephi *ValueIsNewPHI(uint64_t Val, jmorseupdater *Updater) {
    jmorsephi *PHI = ValueIsPHI(Val, Updater);
    if (PHI && PHI->vec.size() == 0)
      return PHI;
    return nullptr;
  }

  /// GetPHIValue - For the specified PHI instruction, return the register
  /// that it defines.
  static uint64_t GetPHIValue(jmorsephi *PHI) {
    return PHI->theval;
  }
};

} // end namespace llvm

void LiveDebugValues::do_the_re_ssaifying_dance(MachineFunction &MF, uint64_t **MLiveIns, uint64_t **MLiveOuts) {
  if (MF.DeSSAdPHIs.empty())
    return;

  // Whip out DeSSAd phis and process them in reverse order: there's a risk
  // that an early PHI depends on a later one that was dessa'd.
  std::vector<DebugInstrRefID> deld;
  for (auto &DeSSA : MF.DeSSAdPHIs)
    deld.push_back(DeSSA.first);
  auto Cmp = [&](const DebugInstrRefID &A, const DebugInstrRefID & B) -> bool {
    return A.getInstID() > B.getInstID();
  };
  llvm::sort(deld, Cmp);

  for (auto &ID : deld) {
    auto &subids = MF.DeSSAdPHIs[ID];
    assert(!subids.empty());
    auto it = SeenInstrIDs.find(ID.getInstID());
    if (it == SeenInstrIDs.end())
      continue;

    // Separate ID numbers out into PHI defs and position defs.
    std::vector<std::pair<DebugInstrRefID, ValueIDNum>> defs, phis;
    std::vector<std::pair<MachineBasicBlock *, ValueIDNum>> blockpos;
    auto deit = MF.DeSSAdPHIs.find(ID);
    for (auto &p : deit->second) {
      auto defit = MF.DeSSAdDefs.find(p);
      if (defit != MF.DeSSAdDefs.end()) {
        // Look up what value we actually read at this point.
        LocIdx L = MTracker->MOToLocIdx(defit->second);
        auto readp = DebugReadPoints.find(p.getInstID());

        // Did this instruction / location disappear? If so, ignore it. This is
        // safe because an undominated predecessor will become an undef input
        // to a phi node.
        if (readp == DebugReadPoints.end())
          continue;

        ValueIDNum val = {0, 0, LocIdx(0)};
        if (L < readp->second.second.size()) {
          val = readp->second.second[L];
          if (val.InstNo == 0)
            // Read from live-ins pls,
            val = ValueIDNum::fromU64(MLiveIns[readp->second.first->getParent()->getNumber()][val.LocNo]);
        }
        // else: When we ran across this instruction, there was no definition
        // for the designated locidx.
        // XXX XXX XXX do we need to force a read?
        defs.push_back(std::make_pair(p, val));
        blockpos.push_back(std::make_pair(readp->second.first->getParent(), val));
      } else {
        auto mit = MF.PHIPointToReg.find(p);

        // Another case of "the phi was here, but was optimised away". We would
        // need to recover the phis earlier. Safe to ignore, because this will
        // be seen as an undef input by the ssa builder if someone reads it.
        if (mit == MF.PHIPointToReg.end())
          continue;

        unsigned bbno = mit->second.first->getNumber();
        LocIdx L = MTracker->MOToLocIdx(mit->second.second);
        ValueIDNum val = {0, 0, LocIdx(0)};
        if (L != 0)
          val = ValueIDNum::fromU64(MLiveIns[bbno][L]);
          
        phis.push_back(std::make_pair(p, val));
        blockpos.push_back(std::make_pair(mit->second.first, val));
      }
    }

    // OK, we've got a set of defs and their locations-ish. And we have
    // locations where they're read. Soooooooooo we can rewrite things
    // using the SSA updater, right?
    jmorseupdater j;
    typedef DenseMap<jmorseblock *, uint64_t> AvailableValsTy;
    AvailableValsTy avail;
    SmallVector<jmorsephi *, 8> created_phis;


    // Define all the available values; rewrite all the uses; then examine
    // all the phis created to see how the rewritten uses map onto available
    // values.
    for (auto &bp : blockpos) {
      avail.insert(std::make_pair(j.getjmorsebb(bp.first), bp.second.asU64()));
    }

    std::vector<std::pair<MachineInstr *, uint64_t>> remapped_values;
    for (auto &lala : it->second) {
      const auto &is_avail = avail.find(j.getjmorsebb(lala->getParent()));
      if (is_avail != avail.end()) { // already got a loc.
        remapped_values.push_back(std::make_pair(lala, is_avail->second));
        continue;
      }
      // For each using inst, get value for this block.
      SSAUpdaterImpl<jmorseupdater> Impl(&j, &avail, &created_phis);
      uint64_t val = Impl.GetValue(j.getjmorsebb(lala->getParent()));
      remapped_values.push_back(std::make_pair(lala, val));
    }

    std::map<jmorsephi *, ValueIDNum> phis_to_value_nums;

    std::map<jmorseblock*, ValueIDNum> block_to_value_map;
    for (auto &bp : blockpos) {
      block_to_value_map[j.getjmorsebb(bp.first)] =  bp.second;
    }

    auto resolve_phi = [&](jmorsephi *jp) -> bool {
      // early out if this is a repeat.
      if (block_to_value_map.find(jp->parent) != block_to_value_map.end())
        return true;

      // Are all these things actually defined?
      for (auto &it : jp->vec) {
        if (j.undef_map.find(&it.first->BB) != j.undef_map.end())
          return false; // reads undef -> die
        if (block_to_value_map.find(it.first) == block_to_value_map.end())
          return false;
      }

      // OK, we have values in all the parent blocks. Now check to see whether
      // they actually form a PHI somewhere, or not.
      // Find locations of outgoing value in one block.
      auto *firstblock = jp->vec[0].first;
      ValueIDNum firstblock_value = block_to_value_map[firstblock];
      uint64_t *firstblock_liveouts = MLiveOuts[firstblock->BB.getNumber()];
      SmallVector<LocIdx, 8> livelocs;
      for (unsigned I = 0; I < MTracker->getNumLocs(); ++I)
        if (firstblock_liveouts[LocIdx(I)] == firstblock_value.asU64())
          livelocs.push_back(LocIdx(I));

      // Alright; how about all the other blocks?
      SmallVector<bool, 8> isvalid;
      isvalid.resize(livelocs.size());
      for (unsigned I = 0; I < isvalid.size(); ++I)
        isvalid[I] = true;

      for (auto &p : jp->vec) {
        uint64_t *block_liveouts = MLiveOuts[p.first->BB.getNumber()];
        ValueIDNum block_val = block_to_value_map[p.first];
        for (unsigned I = 0; I < isvalid.size(); ++I) {
          if (block_liveouts[livelocs[I]] != block_val.asU64())
            isvalid[I] = false;
        }
      }

      unsigned num_valid = 0;
      ValueIDNum res = {0, 0, LocIdx(0)};
      for (unsigned I = 0; I < isvalid.size(); ++I) {
        if (isvalid[I]) {
          ++num_valid;
          res = ValueIDNum::fromU64(MLiveIns[jp->parent->BB.getNumber()][livelocs[I]]);
        }
      }

      //assert(num_valid <= 1 && "Rats, need to resolve multiple phi locs?");
      // This is a pain: and some inputs really are hitting this. Rather than
      // fixing now, just implicitly pick the last one, and risk dropping
      // more locations than necessary.

      if (num_valid == 0)
        // There is no PHI, Neo
        return true;

      // Success.
      block_to_value_map[jp->parent] = res;
      return true;
    };

    unsigned num_remapped = block_to_value_map.size();
    // Now, there's probably a clever way of doing this, but I'm really bored
    // and can't be bothered to implement it. So just repeatedly try and
    // recover all of the phis until we can't recover any more.
    do {
      num_remapped = block_to_value_map.size();
      for (auto &p : created_phis) {
        resolve_phi(p);
      }
    } while (num_remapped != block_to_value_map.size());

    // OK, we've remapped these values. Replace any dbg instr users inside the
    // available blocks with the ValueIDNum read from where the phi was.
    for (auto &p : remapped_values) {
      auto bit = block_to_value_map.find(j.getjmorsebb(p.first->getParent()));
      if (bit == block_to_value_map.end())
        continue; // no resolution.
      assert(bit->second.BlockNo < 0xFFFF); // har.
      dephid_instr_resolutions[p.first] = bit->second;
    }
  }
}

/// Calculate the liveness information for the given machine function and
/// extend ranges across basic blocks.
bool LiveDebugValues::ExtendRanges(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "\nDebug Range Extension\n");

  SmallVector<MLocTransferMap, 32> MLocTransfer;
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

  initialSetup(MF);

  produceMLocTransferFunction(MF, MLocTransfer, MaxNumBlocks);

  // Allocate and initialize two array-of-arrays for the live-in and live-out
  // machine values. The outer dimension is the block number; while the inner
  // dimension is a LocIdx from MLocTracker.
  uint64_t **MOutLocs = new uint64_t *[MaxNumBlocks];
  uint64_t **MInLocs = new uint64_t *[MaxNumBlocks];
  unsigned NumLocs = MTracker->getNumLocs();
  for (int i = 0; i < MaxNumBlocks; ++i) {
    MOutLocs[i] = new uint64_t[NumLocs];
    memset(MOutLocs[i], 0, sizeof(uint64_t) * NumLocs);
    MInLocs[i] = new uint64_t[NumLocs];
    memset(MInLocs[i], 0, sizeof(uint64_t) * NumLocs);
  }

  // Solve the machine value dataflow problem using the MLocTransfer function,
  // storing the computed live-ins / live-outs into the array-of-arrays. We use
  // both live-ins and live-outs for decision making in the variable value
  // dataflow problem.
  mlocDataflow(MInLocs, MOutLocs, MLocTransfer);

  do_the_re_ssaifying_dance(MF, MInLocs, MOutLocs);

  // Walk back through each block / instruction, collecting DBG_VALUE
  // instructions and recording what machine value their operands refer to.
  for (unsigned int I = 0; I < OrderToBB.size(); ++I) {
    MachineBasicBlock &MBB = *OrderToBB[I];
    CurBB = MBB.getNumber();
    VTracker = &vlocs[CurBB];
    VTracker->MBB = &MBB;
    MTracker->loadFromArray(MInLocs[CurBB], CurBB);
    CurInst = 1;
    for (auto &MI : MBB) {
      process(MI, MInLocs);

      // Read abi def regs, for things like loading values out of return value
      // physregs.
      uint64_t instid = MI.peekDebugValueID();
      auto it = MF.ABIRegDef.find(instid);
      if (it != MF.ABIRegDef.end()) {
        // Read the mapped regs and store them for later.
        for (Register Reg : it->second) {
          ValueIDNum ID = MTracker->readReg(Reg);
          DebugInstrRefID NewID(instid, Reg);
          //assert(abimap_regs.find(NewID) == abimap_regs.end());
          // XXX it appears multiple times.
          abimap_regs[NewID] = ID;
        }
      }

      ++CurInst;
    }
    MTracker->reset();
  }

  // Number all variables in the order that they appear, to be used as a stable
  // insertion order later.
  DenseMap<DebugVariable, unsigned> AllVarsNumbering;

  // Map from one LexicalScope to all the variables in that scope.
  DenseMap<const LexicalScope *, SmallSet<DebugVariable, 4>> ScopeToVars;

  // Map from One lexical scope to all blocks in that scope.
  DenseMap<const LexicalScope *, SmallPtrSet<MachineBasicBlock *, 4>>
      ScopeToBlocks;

  // Store a DILocation that describes a scope.
  DenseMap<const LexicalScope *, const DILocation *> ScopeToDILocation;

  // To mirror old LiveDebugValues, enumerate variables in RPOT order. Otherwise
  // the order is unimportant, it just has to be stable.
  for (unsigned int I = 0; I < OrderToBB.size(); ++I) {
    auto *MBB = OrderToBB[I];
    auto *VTracker = &vlocs[MBB->getNumber()];
    // Collect each variable with a DBG_VALUE in this block.
    for (auto &idx : VTracker->Vars) {
      const auto &Var = idx.first;
      const DILocation *ScopeLoc = VTracker->Scopes[Var];
      assert(ScopeLoc != nullptr);
      auto *Scope = LS.findLexicalScope(ScopeLoc);

      // No insts in scope -> shouldn't have been recorded.
      assert(Scope != nullptr);

      AllVarsNumbering.insert(std::make_pair(Var, AllVarsNumbering.size()));
      ScopeToVars[Scope].insert(Var);
      ScopeToBlocks[Scope].insert(VTracker->MBB);
      ScopeToDILocation[Scope] = ScopeLoc;
    }
  }

  // OK. Iterate over scopes: there might be something to be said for
  // ordering them by size/locality, but that's for the future. For each scope,
  // solve the variable value problem, producing a map of variables to values
  // in SavedLiveIns.
  for (auto &P : ScopeToVars) {
    vlocDataflow(P.first, ScopeToDILocation[P.first], P.second,
                 ScopeToBlocks[P.first], SavedLiveIns, MOutLocs, MInLocs,
                 vlocs);
  }

  // Using the computed value locations and variable values for each block,
  // create the DBG_VALUE instructions representing the extended variable
  // locations.
  emitLocations(MF, SavedLiveIns, MInLocs, AllVarsNumbering);

  for (int Idx = 0; Idx < MaxNumBlocks; ++Idx) {
    delete[] MOutLocs[Idx];
    delete[] MInLocs[Idx];
  }
  delete[] MOutLocs;
  delete[] MInLocs;

  // Did we actually make any changes? If we created any DBG_VALUEs, then yes.
  return TTracker->Transfers.size() != 0;
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
  MFI = &MF.getFrameInfo();
  LS.initialize(MF);

  MTracker = new MLocTracker(MF, *TII, *TRI, *MF.getSubtarget().getTargetLowering(), *TFI, *MFI);
  VTracker = nullptr;
  TTracker = nullptr;

  bool Changed = ExtendRanges(MF);
  delete MTracker;
  VTracker = nullptr;
  TTracker = nullptr;

  ArtificialBlocks.clear();
  OrderToBB.clear();
  BBToOrder.clear();
  BBNumToRPO.clear();

  InstrIDMap.clear();
  abimap_regs.clear();
  SeenInstrIDs.clear();
  DebugReadPoints.clear();

  return Changed;
}
