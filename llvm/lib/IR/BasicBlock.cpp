//===-- BasicBlock.cpp - Implement BasicBlock related methods -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the BasicBlock class for the IR library.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/BasicBlock.h"
#include "SymbolTableListTraitsImpl.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "ir"
STATISTIC(NumInstrRenumberings, "Number of renumberings across all blocks");

cl::opt<bool> DDDInhaleDbgValues("experimental-debuginfo-iterators",
                  cl::desc("Enable communicating debuginfo positions through iterators, eliminating intrinsics"),
                  cl::init(false));


DPMarker *BasicBlock::createMarker(Instruction *I) {
  assert(IsInhaled && "Tried to create a marker in a non-inhaled block!");
  assert(I->DbgMarker == nullptr &&
         "Tried to create marker for instuction that already has one!");
  DPMarker *Marker = new DPMarker();
  Marker->MarkedInstr = I;
  I->DbgMarker = Marker;
  return Marker;
}

void BasicBlock::inhaleDbgValues() {
  if (!DDDInhaleDbgValues)
    return;

  IsInhaled = true;
  SmallVector<DPValue *, 4> DPVals;
  for (Instruction &I : make_early_inc_range(InstList)) {
    assert(!I.DbgMarker && "DbgMarker already set on inhalation");
    if (DbgValueInst *DVI = dyn_cast<DbgValueInst>(&I)) {
      DPValue *Value = new DPValue(DVI);
      DPVals.push_back(Value);
      DVI->eraseFromParent();
      continue;
    }
    // Create the marker function that notes this Instruction's position in the DebugValueList.
    createMarker(&I);
    DPMarker *Marker = I.DbgMarker;

    for (DPValue *DPV : DPVals) {
      Marker->insertDPValue(DPV, false);
      DPV->setMarker(Marker);
    }
    DPVals.clear();
  }
}

void BasicBlock::exhaleDbgValues() {
  invalidateOrders();
  IsInhaled = false;
  InstListType::iterator InsertPoint = InstList.begin();
  for (auto &Inst : *this) {
    if (!Inst.DbgMarker)
      continue;

    DPMarker &Marker = *Inst.DbgMarker;
    for (DPValue &DPV : Marker.getDbgValueRange()) {
      InstList.insert(Inst.getIterator(), DPV.createDebugIntrinsic(getModule(), nullptr));
    }

    Marker.eraseFromParent();
  };

  // Assume no danglers?
  assert(TrailingDPValues.StoredDPValues.empty());

  for (auto &I : *this)
    assert(!I.DbgMarker);
}

void BasicBlock::validateDbgValues() {
  // Only validate if we have a debug program.
  if (!IsInhaled)
    return;

  // Match every DebugProgramMarker to every Instruction and vice versa, and
  // verify that there are no invalid DebugProgramValues.
  for (auto BlockInstructionIt = begin(); BlockInstructionIt != end(); ++BlockInstructionIt) {
    if (!BlockInstructionIt->DbgMarker)
      continue;

    // Validate DebugProgramMarkers.
    DPMarker *CurrentDebugMarker = BlockInstructionIt->DbgMarker;

    // If this is a marker, it should match the instruction and vice versa.
    assert(CurrentDebugMarker->MarkedInstr == &*BlockInstructionIt &&
           "Debug Marker points to incorrect instruction?");

    // Now validate any DPValues in the marker.
    for (DPValue &DPV : CurrentDebugMarker->getDbgValueRange()) {
      // Validate DebugProgramValues.
      assert(DPV.getMarker() == CurrentDebugMarker && "Not pointing at correct next marker!");

      // Verify that no DbgValues appear prior to PHIs.
      assert(!isa<PHINode>(BlockInstructionIt) &&
             "DebugProgramValues must not appear before PHI nodes in a block!");
    }
  }
}

void BasicBlock::setInhaled(bool NewInhaled) {
  if (NewInhaled && !IsInhaled)
    inhaleDbgValues();
  else if (!NewInhaled && IsInhaled)
    exhaleDbgValues();
}

#ifndef NDEBUG
void BasicBlock::dumpDbgValues() const {
  for (auto &Inst : *this) {
    if (!Inst.DbgMarker)
      continue;

    dbgs() << "@ " << Inst.DbgMarker << " ";
    Inst.DbgMarker->dump();
  };
}
#endif

ValueSymbolTable *BasicBlock::getValueSymbolTable() {
  if (Function *F = getParent())
    return F->getValueSymbolTable();
  return nullptr;
}

LLVMContext &BasicBlock::getContext() const {
  return getType()->getContext();
}

template <> void llvm::invalidateParentIListOrdering(BasicBlock *BB) {
  BB->invalidateOrders();
}

// Explicit instantiation of SymbolTableListTraits since some of the methods
// are not in the public header file...
template class llvm::SymbolTableListTraits<Instruction>;

BasicBlock::BasicBlock(LLVMContext &C, const Twine &Name, Function *NewParent,
                       BasicBlock *InsertBefore)
  : Value(Type::getLabelTy(C), Value::BasicBlockVal), Parent(nullptr) {

  IsInhaled = false;
  if (NewParent)
    insertInto(NewParent, InsertBefore);
  else
    assert(!InsertBefore &&
           "Cannot insert block before another block with no function!");

  setName(Name);
  if (NewParent)
    setInhaled(NewParent->IsInhaled);
}

void BasicBlock::insertInto(Function *NewParent, BasicBlock *InsertBefore) {
  assert(NewParent && "Expected a parent");
  assert(!Parent && "Already has a parent");

  setInhaled(NewParent->IsInhaled);

  if (InsertBefore)
    NewParent->insert(InsertBefore->getIterator(), this);
  else
    NewParent->insert(NewParent->end(), this);
  if (NewParent->IsInhaled && !IsInhaled)
    inhaleDbgValues();
  else if (!NewParent->IsInhaled && IsInhaled)
    exhaleDbgValues();
}

void BasicBlock::insertInto(Function *NewParent,
                            Function::iterator InsertBefore) {
  assert(NewParent && "Expected a parent");
  assert(!Parent && "Already has a parent");

  setInhaled(NewParent->IsInhaled);
  NewParent->insert(InsertBefore, this);
}

BasicBlock::~BasicBlock() {
  validateInstrOrdering();

  // If the address of the block is taken and it is being deleted (e.g. because
  // it is dead), this means that there is either a dangling constant expr
  // hanging off the block, or an undefined use of the block (source code
  // expecting the address of a label to keep the block alive even though there
  // is no indirect branch).  Handle these cases by zapping the BlockAddress
  // nodes.  There are no other possible uses at this point.
  if (hasAddressTaken()) {
    assert(!use_empty() && "There should be at least one blockaddress!");
    Constant *Replacement =
      ConstantInt::get(llvm::Type::getInt32Ty(getContext()), 1);
    while (!use_empty()) {
      BlockAddress *BA = cast<BlockAddress>(user_back());
      BA->replaceAllUsesWith(ConstantExpr::getIntToPtr(Replacement,
                                                       BA->getType()));
      BA->destroyConstant();
    }
  }

  assert(getParent() == nullptr && "BasicBlock still linked into the program!");
  dropAllReferences();
  for (auto &Inst : *this) {
    if (!Inst.DbgMarker)
      continue;
    Inst.DbgMarker->eraseFromParent();
  }
  InstList.clear();
}

void BasicBlock::setParent(Function *parent) {
  // Set Parent=parent, updating instruction symtab entries as appropriate.
  InstList.setSymTabObject(&Parent, parent);
}

iterator_range<filter_iterator<BasicBlock::const_iterator,
                               std::function<bool(const Instruction &)>>>
BasicBlock::instructionsWithoutDebug(bool SkipPseudoOp) const {
  std::function<bool(const Instruction &)> Fn = [=](const Instruction &I) {
    return !isa<DbgInfoIntrinsic>(I) &&
           !(SkipPseudoOp && isa<PseudoProbeInst>(I));
  };
  return make_filter_range(*this, Fn);
}

iterator_range<
    filter_iterator<BasicBlock::iterator, std::function<bool(Instruction &)>>>
BasicBlock::instructionsWithoutDebug(bool SkipPseudoOp) {
  std::function<bool(Instruction &)> Fn = [=](Instruction &I) {
    return !isa<DbgInfoIntrinsic>(I) &&
           !(SkipPseudoOp && isa<PseudoProbeInst>(I));
  };
  return make_filter_range(*this, Fn);
}

filter_iterator<BasicBlock::const_iterator,
                std::function<bool(const Instruction &)>>::difference_type
BasicBlock::sizeWithoutDebug() const {
  return std::distance(instructionsWithoutDebug().begin(),
                       instructionsWithoutDebug().end());
}

void BasicBlock::removeFromParent() {
  getParent()->getBasicBlockList().remove(getIterator());
}

iplist<BasicBlock>::iterator BasicBlock::eraseFromParent() {
  return getParent()->getBasicBlockList().erase(getIterator());
}

void BasicBlock::moveBefore(BasicBlock *MovePos) {
  MovePos->getParent()->splice(MovePos->getIterator(), getParent(),
                               getIterator());
}

void BasicBlock::moveAfter(BasicBlock *MovePos) {
  MovePos->getParent()->splice(++MovePos->getIterator(), getParent(),
                               getIterator());
}

const Module *BasicBlock::getModule() const {
  return getParent()->getParent();
}

const CallInst *BasicBlock::getTerminatingMustTailCall() const {
  if (InstList.empty())
    return nullptr;
  const ReturnInst *RI = dyn_cast<ReturnInst>(&InstList.back());
  if (!RI || RI == &InstList.front())
    return nullptr;

  const Instruction *Prev = RI->getPrevNode();
  if (!Prev)
    return nullptr;

  if (Value *RV = RI->getReturnValue()) {
    if (RV != Prev)
      return nullptr;

    // Look through the optional bitcast.
    if (auto *BI = dyn_cast<BitCastInst>(Prev)) {
      RV = BI->getOperand(0);
      Prev = BI->getPrevNode();
      if (!Prev || RV != Prev)
        return nullptr;
    }
  }

  if (auto *CI = dyn_cast<CallInst>(Prev)) {
    if (CI->isMustTailCall())
      return CI;
  }
  return nullptr;
}

const CallInst *BasicBlock::getTerminatingDeoptimizeCall() const {
  if (InstList.empty())
    return nullptr;
  auto *RI = dyn_cast<ReturnInst>(&InstList.back());
  if (!RI || RI == &InstList.front())
    return nullptr;

  if (auto *CI = dyn_cast_or_null<CallInst>(RI->getPrevNode()))
    if (Function *F = CI->getCalledFunction())
      if (F->getIntrinsicID() == Intrinsic::experimental_deoptimize)
        return CI;

  return nullptr;
}

const CallInst *BasicBlock::getPostdominatingDeoptimizeCall() const {
  const BasicBlock* BB = this;
  SmallPtrSet<const BasicBlock *, 8> Visited;
  Visited.insert(BB);
  while (auto *Succ = BB->getUniqueSuccessor()) {
    if (!Visited.insert(Succ).second)
      return nullptr;
    BB = Succ;
  }
  return BB->getTerminatingDeoptimizeCall();
}

const Instruction *BasicBlock::getFirstMayFaultInst() const {
  if (InstList.empty())
    return nullptr;
  for (const Instruction &I : *this)
    if (isa<LoadInst>(I) || isa<StoreInst>(I) || isa<CallBase>(I))
      return &I;
  return nullptr;
}

const Instruction* BasicBlock::getFirstNonPHI() const {
  for (const Instruction &I : *this)
    if (!isa<PHINode>(I))
      return &I;
  return nullptr;
}

const Instruction *BasicBlock::getFirstNonPHIOrDbg(bool SkipPseudoOp) const {
  for (const Instruction &I : *this) {
    if (isa<PHINode>(I) || isa<DbgInfoIntrinsic>(I))
      continue;

    if (SkipPseudoOp && isa<PseudoProbeInst>(I))
      continue;

    return &I;
  }
  return nullptr;
}

const Instruction *
BasicBlock::getFirstNonPHIOrDbgOrLifetime(bool SkipPseudoOp) const {
  for (const Instruction &I : *this) {
    if (isa<PHINode>(I) || isa<DbgInfoIntrinsic>(I))
      continue;

    if (I.isLifetimeStartOrEnd())
      continue;

    if (SkipPseudoOp && isa<PseudoProbeInst>(I))
      continue;

    return &I;
  }
  return nullptr;
}

BasicBlock::const_iterator BasicBlock::getFirstInsertionPt() const {
  const Instruction *FirstNonPHI = getFirstNonPHI();
  if (!FirstNonPHI)
    return end();

  const_iterator InsertPt = FirstNonPHI->getIterator();
  if (InsertPt->isEHPad()) ++InsertPt;
  // Signal to users of this iterator that it's supposed to come "before" any
  // debug-info at the start of the block.
  InsertPt.setHeadBit(true);
  return InsertPt;
}

BasicBlock::const_iterator BasicBlock::getFirstNonPHIOrDbgOrAlloca() const {
  const Instruction *FirstNonPHI = getFirstNonPHI();
  if (!FirstNonPHI)
    return end();

  const_iterator InsertPt = FirstNonPHI->getIterator();
  if (InsertPt->isEHPad())
    ++InsertPt;

  if (isEntryBlock()) {
    const_iterator End = end();
    while (InsertPt != End &&
           (isa<AllocaInst>(*InsertPt) || isa<DbgInfoIntrinsic>(*InsertPt) ||
            isa<PseudoProbeInst>(*InsertPt))) {
      if (const AllocaInst *AI = dyn_cast<AllocaInst>(&*InsertPt)) {
        if (!AI->isStaticAlloca())
          break;
      }
      ++InsertPt;
    }
  }
  return InsertPt;
}

void BasicBlock::dropAllReferences() {
  for (Instruction &I : *this)
    I.dropAllReferences();
}

const BasicBlock *BasicBlock::getSinglePredecessor() const {
  const_pred_iterator PI = pred_begin(this), E = pred_end(this);
  if (PI == E) return nullptr;         // No preds.
  const BasicBlock *ThePred = *PI;
  ++PI;
  return (PI == E) ? ThePred : nullptr /*multiple preds*/;
}

const BasicBlock *BasicBlock::getUniquePredecessor() const {
  const_pred_iterator PI = pred_begin(this), E = pred_end(this);
  if (PI == E) return nullptr; // No preds.
  const BasicBlock *PredBB = *PI;
  ++PI;
  for (;PI != E; ++PI) {
    if (*PI != PredBB)
      return nullptr;
    // The same predecessor appears multiple times in the predecessor list.
    // This is OK.
  }
  return PredBB;
}

bool BasicBlock::hasNPredecessors(unsigned N) const {
  return hasNItems(pred_begin(this), pred_end(this), N);
}

bool BasicBlock::hasNPredecessorsOrMore(unsigned N) const {
  return hasNItemsOrMore(pred_begin(this), pred_end(this), N);
}

const BasicBlock *BasicBlock::getSingleSuccessor() const {
  const_succ_iterator SI = succ_begin(this), E = succ_end(this);
  if (SI == E) return nullptr; // no successors
  const BasicBlock *TheSucc = *SI;
  ++SI;
  return (SI == E) ? TheSucc : nullptr /* multiple successors */;
}

const BasicBlock *BasicBlock::getUniqueSuccessor() const {
  const_succ_iterator SI = succ_begin(this), E = succ_end(this);
  if (SI == E) return nullptr; // No successors
  const BasicBlock *SuccBB = *SI;
  ++SI;
  for (;SI != E; ++SI) {
    if (*SI != SuccBB)
      return nullptr;
    // The same successor appears multiple times in the successor list.
    // This is OK.
  }
  return SuccBB;
}

iterator_range<BasicBlock::phi_iterator> BasicBlock::phis() {
  PHINode *P = empty() ? nullptr : dyn_cast<PHINode>(&*begin());
  return make_range<phi_iterator>(P, nullptr);
}

void BasicBlock::removePredecessor(BasicBlock *Pred,
                                   bool KeepOneInputPHIs) {
  // Use hasNUsesOrMore to bound the cost of this assertion for complex CFGs.
  assert((hasNUsesOrMore(16) || llvm::is_contained(predecessors(this), Pred)) &&
         "Pred is not a predecessor!");

  // Return early if there are no PHI nodes to update.
  if (empty() || !isa<PHINode>(begin()))
    return;

  unsigned NumPreds = cast<PHINode>(front()).getNumIncomingValues();
  for (PHINode &Phi : make_early_inc_range(phis())) {
    Phi.removeIncomingValue(Pred, !KeepOneInputPHIs);
    if (KeepOneInputPHIs)
      continue;

    // If we have a single predecessor, removeIncomingValue may have erased the
    // PHI node itself.
    if (NumPreds == 1)
      continue;

    // Try to replace the PHI node with a constant value.
    if (Value *PhiConstant = Phi.hasConstantValue()) {
      Phi.replaceAllUsesWith(PhiConstant);
      Phi.eraseFromParent();
    }
  }
}

bool BasicBlock::canSplitPredecessors() const {
  const Instruction *FirstNonPHI = getFirstNonPHI();
  if (isa<LandingPadInst>(FirstNonPHI))
    return true;
  // This is perhaps a little conservative because constructs like
  // CleanupBlockInst are pretty easy to split.  However, SplitBlockPredecessors
  // cannot handle such things just yet.
  if (FirstNonPHI->isEHPad())
    return false;
  return true;
}

bool BasicBlock::isLegalToHoistInto() const {
  auto *Term = getTerminator();
  // No terminator means the block is under construction.
  if (!Term)
    return true;

  // If the block has no successors, there can be no instructions to hoist.
  assert(Term->getNumSuccessors() > 0);

  // Instructions should not be hoisted across exception handling boundaries.
  return !Term->isExceptionalTerminator();
}

bool BasicBlock::isEntryBlock() const {
  const Function *F = getParent();
  assert(F && "Block must have a parent function to use this API");
  return this == &F->getEntryBlock();
}

BasicBlock *BasicBlock::splitBasicBlock(iterator I, const Twine &BBName,
                                        bool Before) {
  if (Before)
    return splitBasicBlockBefore(I, BBName);

  assert(getTerminator() && "Can't use splitBasicBlock on degenerate BB!");
  assert(I != InstList.end() &&
         "Trying to get me to create degenerate basic block!");

  BasicBlock *New = BasicBlock::Create(getContext(), BBName, getParent(),
                                       this->getNextNode());

  // Save DebugLoc of split point before invalidating iterator.
  DebugLoc Loc = I->getStableDebugLoc();
  // Move all of the specified instructions from the original basic block into
  // the new basic block.
  New->splice(New->end(), this, I, end());

  // Add a branch instruction to the newly formed basic block.
  BranchInst *BI = BranchInst::Create(New, this);
  BI->setDebugLoc(Loc);

  // Now we must loop through all of the successors of the New block (which
  // _were_ the successors of the 'this' block), and update any PHI nodes in
  // successors.  If there were PHI nodes in the successors, then they need to
  // know that incoming branches will be from New, not from Old (this).
  //
  New->replaceSuccessorsPhiUsesWith(this, New);
  return New;
}

BasicBlock *BasicBlock::splitBasicBlockBefore(iterator I, const Twine &BBName) {
  assert(getTerminator() &&
         "Can't use splitBasicBlockBefore on degenerate BB!");
  assert(I != InstList.end() &&
         "Trying to get me to create degenerate basic block!");

  assert((!isa<PHINode>(*I) || getSinglePredecessor()) &&
         "cannot split on multi incoming phis");

  BasicBlock *New = BasicBlock::Create(getContext(), BBName, getParent(), this);
  // Save DebugLoc of split point before invalidating iterator.
  DebugLoc Loc = I->getDebugLoc();
  // Move all of the specified instructions from the original basic block into
  // the new basic block.
  New->splice(New->end(), this, begin(), I);

  // Loop through all of the predecessors of the 'this' block (which will be the
  // predecessors of the New block), replace the specified successor 'this'
  // block to point at the New block and update any PHI nodes in 'this' block.
  // If there were PHI nodes in 'this' block, the PHI nodes are updated
  // to reflect that the incoming branches will be from the New block and not
  // from predecessors of the 'this' block.
  // Save predecessors to separate vector before modifying them.
  SmallVector<BasicBlock *, 4> Predecessors;
  for (BasicBlock *Pred : predecessors(this))
    Predecessors.push_back(Pred);
  for (BasicBlock *Pred : Predecessors) {
    Instruction *TI = Pred->getTerminator();
    TI->replaceSuccessorWith(this, New);
    this->replacePhiUsesWith(Pred, New);
  }
  // Add a branch instruction from  "New" to "this" Block.
  BranchInst *BI = BranchInst::Create(this, New);
  BI->setDebugLoc(Loc);

  return New;
}

void BasicBlock::splice(BasicBlock::iterator ToIt, BasicBlock *FromBB,
                        BasicBlock::iterator FromBeginIt,
                        BasicBlock::iterator FromEndIt) {
#ifdef EXPENSIVE_CHECKS
  // Check that FromBeginIt is befor FromEndIt.
  auto FromBBEnd = FromBB->end();
  for (auto It = FromBeginIt; It != FromEndIt; ++It)
    assert(It != FromBBEnd && "FromBeginIt not before FromEndIt!");
#endif // EXPENSIVE_CHECKS
  getInstList().splice(ToIt, FromBB->getInstList(), FromBeginIt, FromEndIt);
}

BasicBlock::iterator BasicBlock::erase(BasicBlock::iterator FromIt,
                                       BasicBlock::iterator ToIt) {
  return InstList.erase(FromIt, ToIt);
}

void BasicBlock::replacePhiUsesWith(BasicBlock *Old, BasicBlock *New) {
  // N.B. This might not be a complete BasicBlock, so don't assume
  // that it ends with a non-phi instruction.
  for (Instruction &I : *this) {
    PHINode *PN = dyn_cast<PHINode>(&I);
    if (!PN)
      break;
    PN->replaceIncomingBlockWith(Old, New);
  }
}

void BasicBlock::replaceSuccessorsPhiUsesWith(BasicBlock *Old,
                                              BasicBlock *New) {
  Instruction *TI = getTerminator();
  if (!TI)
    // Cope with being called on a BasicBlock that doesn't have a terminator
    // yet. Clang's CodeGenFunction::EmitReturnBlock() likes to do this.
    return;
  for (BasicBlock *Succ : successors(TI))
    Succ->replacePhiUsesWith(Old, New);
}

void BasicBlock::replaceSuccessorsPhiUsesWith(BasicBlock *New) {
  this->replaceSuccessorsPhiUsesWith(this, New);
}

bool BasicBlock::isLandingPad() const {
  return isa<LandingPadInst>(getFirstNonPHI());
}

const LandingPadInst *BasicBlock::getLandingPadInst() const {
  return dyn_cast<LandingPadInst>(getFirstNonPHI());
}

std::optional<uint64_t> BasicBlock::getIrrLoopHeaderWeight() const {
  const Instruction *TI = getTerminator();
  if (MDNode *MDIrrLoopHeader =
      TI->getMetadata(LLVMContext::MD_irr_loop)) {
    MDString *MDName = cast<MDString>(MDIrrLoopHeader->getOperand(0));
    if (MDName->getString().equals("loop_header_weight")) {
      auto *CI = mdconst::extract<ConstantInt>(MDIrrLoopHeader->getOperand(1));
      return std::optional<uint64_t>(CI->getValue().getZExtValue());
    }
  }
  return std::nullopt;
}

BasicBlock::iterator llvm::skipDebugIntrinsics(BasicBlock::iterator It) {
  while (isa<DbgInfoIntrinsic>(It))
    ++It;
  return It;
}

void BasicBlock::renumberInstructions() {
  unsigned Order = 0;
  for (Instruction &I : *this)
    I.Order = Order++;

  // Set the bit to indicate that the instruction order valid and cached.
  BasicBlockBits Bits = getBasicBlockBits();
  Bits.InstrOrderValid = true;
  setBasicBlockBits(Bits);

  NumInstrRenumberings++;
}

// XXX this is unused?
auto BasicBlock::getDanglingDbgValues() -> iterator_range<DIIterator> {
  // Return the range of DPValues and other items that are hanging around at
  // the end of the block, past any terminator. This is a non-cannonical form.
  return TrailingDPValues.getDbgValueRange();
}

void BasicBlock::flushTerminatorDbgValues() {
  // If we juggle the terminators in a block, any DPValues will sink and "fall
  // off the end", existing after any terminator that gets inserted. Where
  // normally if we inserted a terminator at the end of a block, it would come
  // after any dbg.values. To get out of this unfortunate form, whenever we
  // insert a terminator, check whether there's anything dangling at the end
  // and move those DPValues in front of the terminator.

  if (!IsInhaled)
    return;

  Instruction *Term = getTerminator();
  if (!Term)
    return;
  
  // Are there any dangling DPValues?
  if (TrailingDPValues.StoredDPValues.empty())
    return;

  // We might have already been at the end anyway... walk backwards through
  // any DPValues resetting the marker.
  Term->DbgMarker->absorbDebugValues(TrailingDPValues, false);
}

void BasicBlock::blockSplice(iterator Dest, BasicBlock *Src, iterator First, iterator Last) {
  // Find out where to _place_ these dbg.values; if InsertAtHead is specified,
  // this will be at the start of Dest's debug value range, otherwise this is
  // just Dest's marker.
  bool InsertAtHead = Dest.getHeadBit();
  bool ReadFromHead = First.getHeadBit();
  // Use this flag to signal the abnormal case, where we don't want to copy the
  // DPValues ahead of the "Last" position. 
  bool ReadFromTail = !Last.getTailBit();

  // Lots of horrible special casing for empty transfers: the dbg.values between
  // two positions could be spliced in dbg.value mode.
  if (First == Last) {
    if (!IsInhaled)
      return;
    if (!Src->empty()) {
      // Non-empty block. Are we copying from the head?
      if (!Src->empty() && First == Src->begin() && ReadFromHead) {
        Dest->DbgMarker->absorbDebugValues(*First->DbgMarker, InsertAtHead);
        return;
      }
    } else {
      // Oh dear; dbg.values in an empty block. Move them over too.
      assert(Dest != end() && "What about dangling off end DPValues?");
      DPMarker *M = Dest->DbgMarker; 
      M->absorbDebugValues(Src->TrailingDPValues, InsertAtHead);
      return;
    }
  }

  assert(Src->IsInhaled == IsInhaled);

  // Debug stuff -- remove marker from Dest.
  DPMarker *DestMarker = nullptr;
  if (IsInhaled && Dest != end()) {
    DestMarker = getMarker(Dest);
    DestMarker->removeFromParent();
    createMarker(&*Dest);
  }

  // If we're moving the tail range of DPValues, re-set their marker to the tail of the new block.
  if (IsInhaled && ReadFromTail) {
    DPMarker *OntoDest = getMarker(Dest);
    DPMarker *FromLast = Src->getMarker(Last);
    OntoDest->absorbDebugValues(*FromLast, true);
  }

  // If we're _not_ reading from head, move their markers onto Last.
  if (IsInhaled && !ReadFromHead) {
    DPMarker *OntoLast = Src->getMarker(Last);
    DPMarker *FromFirst = Src->getMarker(First);
    OntoLast->absorbDebugValues(*FromFirst, true); // Always insert at head of it.
  }

  // Finally, re-apply DPMarker information at head or tail.
  if (DestMarker) {
    if (InsertAtHead) {
      // We put instrs and debug-info ahead of these DPValues, insert at tail of Dest.
      DPMarker *NewDestMarker = getMarker(Dest);
      NewDestMarker->absorbDebugValues(*DestMarker, false);
    } else {
      // We inserted immediately before Dest, so dbg.values will go on the _front_ of First.
      DPMarker *FirstMarker = getMarker(First);
      FirstMarker->absorbDebugValues(*DestMarker, true);
    }
    DestMarker->eraseFromParent();
  } else if (IsInhaled) {
    // We inserted ahead of end(), or not:
    if (InsertAtHead) {
      // Leave dangling DPValues where they are,
    } else {
      // Dangling DPValues go ahead of First now.
      DPMarker *FirstMarker = getMarker(First);
      FirstMarker->absorbDebugValues(TrailingDPValues, true);
    }
  }

  // And move the instructions.
  getInstList().splice(Dest, Src->getInstList(), First, Last);

  flushTerminatorDbgValues();
}

void BasicBlock::blockSplice(iterator Dest, BasicBlock *Src) {
  blockSplice(Dest, Src, Src->begin(), Src->end());
}

void BasicBlock::insertDPValueAfter(DPValue *DPV, Instruction *I) {
  assert(IsInhaled);
  assert(I->getParent() == this);

  iterator NextIt = std::next(I->getIterator());
  DPMarker *NextMarker = (NextIt == end()) ? &TrailingDPValues : NextIt->DbgMarker;
  NextMarker->insertDPValue(DPV, true);
}

void BasicBlock::insertDPValueBefore(DPValue *DPV, InstListType::iterator Where) {
  // Assume that we don't insertBefore BasicBlock::end().
  assert(Where->getParent() == this);
  bool InsertAtHead = Where.getHeadBit();
  Where->DbgMarker->insertDPValue(DPV, InsertAtHead);
}

DPMarker *BasicBlock::getNextMarker(Instruction *I) { // The _next_ marker, either TrailingDPValues or the actual marker
  return getMarker(std::next(I->getIterator()));
}

DPMarker *BasicBlock::getMarker(InstListType::iterator It) { // either TrailingDPValues or the actual marker
  if (It == end())
    return &TrailingDPValues;
  return It->DbgMarker;
}

#ifndef NDEBUG
/// In asserts builds, this checks the numbering. In non-asserts builds, it
/// is defined as a no-op inline function in BasicBlock.h.
void BasicBlock::validateInstrOrdering() const {
  if (!isInstrOrderValid())
    return;
  const Instruction *Prev = nullptr;
  for (const Instruction &I : *this) {
    assert((!Prev || Prev->comesBefore(&I)) &&
           "cached instruction ordering is incorrect");
    Prev = &I;
  }
}
#endif
