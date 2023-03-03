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

using namespace llvm;

#define DEBUG_TYPE "ir"
STATISTIC(NumInstrRenumberings, "Number of renumberings across all blocks");

bool DDDInhaleDbgValues = true;

BasicBlock::DIIterator BasicBlock::createMarker(DIIterator InsertPos, Instruction *I) {
  assert(IsInhaled && "Tried to create a marker in a non-inhaled block!");
  assert(I->DbgMarker == nullptr &&
         "Tried to create marker for instuction that already has one!");
  DPMarker *Marker = new DPMarker();
  Marker->MarkedInstr = I;
  I->DbgMarker = Marker;
  return DbgProgramList.insert(InsertPos, *Marker);
}

void BasicBlock::inhaleDbgValues() {
  if (!DDDInhaleDbgValues)
    return;

  IsInhaled = true;
  assert(DbgProgramList.empty());
  for (Instruction &I : make_early_inc_range(InstList)) {
    assert(!I.DbgMarker && "DbgMarker already set on inhalation");
    if (DbgValueInst *DVI = dyn_cast<DbgValueInst>(&I)) {
      DPValue *Value = new DPValue(DVI);
      DbgProgramList.insert(DbgProgramList.end(), *Value);
      DVI->eraseFromParent();
      continue;
    }
    // Create the marker function that notes this Instruction's position in the DebugValueList.
    createMarker(DbgProgramList.end(), &I);
  }
}

void BasicBlock::exhaleDbgValues() {
  invalidateOrders();
  IsInhaled = false;
  InstListType::iterator InsertPoint = InstList.begin();
  for (DebugProgramInstruction &DPI : DbgProgramList) {
    if (DPI.getInstrType() == DebugProgramInstruction::DPInstrType::Marker) {
      DPMarker *Marker = static_cast<DPMarker*>(&DPI);
      assert(Marker->MarkedInstr == &*InsertPoint);
      Marker->MarkedInstr->DbgMarker = nullptr;
      InsertPoint = std::next(InsertPoint);
      continue;
    }
    assert(DPI.getInstrType() == DebugProgramInstruction::DPInstrType::Value);
    DPValue *DPV = static_cast<DPValue*>(&DPI);
    InstList.insert(InsertPoint, DPV->createDebugIntrinsic(getModule(), nullptr));
  }

  for (auto &DPI : make_early_inc_range(DbgProgramList)) {
    DbgProgramList.erase(DPI.getIterator());
    DPI.deleteInstr();
  }

  for (auto &I : *this)
    assert(!I.DbgMarker);
}

void BasicBlock::validateDbgValues() {
  // Only validate if we have a debug program.
  if (DbgProgramList.empty())
    return;
  assert(!empty() && "BasicBlock with DebugProgram but no Instructions?");

  // Match every DebugProgramMarker to every Instruction and vice versa, and
  // verify that there are no invalid DebugProgramValues.
  auto DbgProgramIt = DbgProgramList.begin();
  auto BlockInstructionIt = begin();
  while (DbgProgramIt != DbgProgramList.end() && BlockInstructionIt != end()) {
    auto *CurrentBlockInstruction = &*BlockInstructionIt;
    auto *CurrentDebugProgramInstruction = &*DbgProgramIt;
    if (DbgProgramIt->isMarker()) {
      // Validate DebugProgramMarkers.
      auto *CurrentDebugMarker =
          static_cast<DPMarker *>(CurrentDebugProgramInstruction);

      // If this is a marker, it should match the instruction and vice versa.
      assert(CurrentDebugMarker->MarkedInstr == CurrentBlockInstruction &&
             "Debug Marker points to incorrect instruction?");
      assert(CurrentBlockInstruction->DbgMarker == CurrentDebugMarker &&
             "Instruction points to incorrect Debug Marker?");

      // Advance both iterators.
      ++DbgProgramIt;
      ++BlockInstructionIt;
    } else {
      assert(DbgProgramIt->getParent() == this);
      // Validate DebugProgramValues.
      assert(CurrentDebugProgramInstruction->isValue() &&
             "If not marker, DebugProgramInstuction must be a value.");
      auto *CurrentDebugValue =
          static_cast<DPValue *>(CurrentDebugProgramInstruction);

      // Verify that no DbgValues appear prior to PHIs.
      assert(!isa<PHINode>(CurrentBlockInstruction) &&
             "DebugProgramValues must not appear before PHI nodes in a block!");

      // Advance only the DebugProgram iterator, as we are "between"
      // instructions.
      ++DbgProgramIt;
    }
  }

  // Both normal and debug programs should end with the terminator and its
  // marker, and so should finish at the same point.
  assert(DbgProgramIt == DbgProgramList.end() &&
         "Dangling Debug Program Instructions after end of block!");
  assert(BlockInstructionIt == end() &&
         "Dangling Instructions after end of Debug Program!");
}

void BasicBlock::setInhaled(bool NewInhaled) {
  if (NewInhaled && !IsInhaled)
    inhaleDbgValues();
  else if (!NewInhaled && IsInhaled)
    exhaleDbgValues();
}

#ifndef NDEBUG
void BasicBlock::dumpDbgValues() const {
  for (auto &DPI : DbgProgramList) {
    dbgs() << "@ " << &DPI << " ";
    DPI.dump();
  }
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

  MarkerFn = nullptr;
  IsInhaled = false;
  if (NewParent)
    insertInto(NewParent, InsertBefore);
  else
    assert(!InsertBefore &&
           "Cannot insert block before another block with no function!");

  setName(Name);
}

void BasicBlock::insertInto(Function *NewParent, BasicBlock *InsertBefore) {
  assert(NewParent && "Expected a parent");
  assert(!Parent && "Already has a parent");

  setInhaled(NewParent->IsInhaled);
  if (InsertBefore)
    NewParent->getBasicBlockList().insert(InsertBefore->getIterator(), this);
  else
    NewParent->getBasicBlockList().push_back(this);
}

void BasicBlock::insertInto(Function *NewParent,
                            Function::iterator InsertBefore) {
  assert(NewParent && "Expected a parent");
  assert(!Parent && "Already has a parent");

  setInhaled(NewParent->IsInhaled);
  NewParent->getBasicBlockList().insert(InsertBefore, this);
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
  InstList.clear();
  for (auto &DPI : make_early_inc_range(DbgProgramList)) {
    DbgProgramList.erase(DPI.getIterator());
    DPI.deleteInstr();
  }
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
  MovePos->getParent()->functionSplice(MovePos->getIterator(), getParent(),
                                       this);
}

void BasicBlock::moveAfter(BasicBlock *MovePos) {
  MovePos->getParent()->functionSplice(++MovePos->getIterator(), getParent(),
                                       this);
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
  DebugLoc Loc = I->getDebugLoc();
  // DDD: Don't take DebugLocs from debug intrinsics.
  if (isa<DbgInfoIntrinsic>(&*I))
    Loc = I->getNextNonDebugInstruction()->getDebugLoc();
  // Move all of the specified instructions from the original basic block into
  // the new basic block.
  New->blockSplice(New->end(), this, I, end());

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
  New->blockSplice(New->end(), this, begin(), I);

  // Loop through all of the predecessors of the 'this' block (which will be the
  // predecessors of the New block), replace the specified successor 'this'
  // block to point at the New block and update any PHI nodes in 'this' block.
  // If there were PHI nodes in 'this' block, the PHI nodes are updated
  // to reflect that the incoming branches will be from the New block and not
  // from predecessors of the 'this' block.
  for (BasicBlock *Pred : predecessors(this)) {
    Instruction *TI = Pred->getTerminator();
    TI->replaceSuccessorWith(this, New);
    this->replacePhiUsesWith(Pred, New);
  }
  // Add a branch instruction from  "New" to "this" Block.
  BranchInst *BI = BranchInst::Create(this, New);
  BI->setDebugLoc(Loc);

  return New;
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

Optional<uint64_t> BasicBlock::getIrrLoopHeaderWeight() const {
  const Instruction *TI = getTerminator();
  if (MDNode *MDIrrLoopHeader =
      TI->getMetadata(LLVMContext::MD_irr_loop)) {
    MDString *MDName = cast<MDString>(MDIrrLoopHeader->getOperand(0));
    if (MDName->getString().equals("loop_header_weight")) {
      auto *CI = mdconst::extract<ConstantInt>(MDIrrLoopHeader->getOperand(1));
      return Optional<uint64_t>(CI->getValue().getZExtValue());
    }
  }
  return Optional<uint64_t>();
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

auto BasicBlock::getDanglingDbgValues() -> iterator_range<DIIterator> {
  // Return the range of DPValues and other items that are hanging around at
  // the end of the block, past any terminator. This is a non-cannonical form.
  return make_range(beginDebugValueRange(InstList.end()), DbgProgramList.end());
}

void BasicBlock::flushTerminatorDbgValues() {
  // If we juggle the terminators in a block, any DPValues will sink and "fall
  // off the end", existing after any terminator that gets inserted. Where
  // normally if we inserted a terminator at the end of a block, it would come
  // after any dbg.values. To get out of this unfortunate form, whenever we
  // insert a terminator, check whether there's anything dangling at the end
  // and move those DPValues in front of the terminator.

  if (DbgProgramList.empty())
    return;

  Instruction *Term = getTerminator();
  if (!Term)
    return;
  
  // Move the terminator's marker to the end of the list. Function is no-op if the terminator's marker is already at the end (i.e. there are no dangling debug values).
  auto MarkerIt = Term->DbgMarker->getIterator();
  DbgProgramList.splice(DbgProgramList.end(), DbgProgramList, MarkerIt, std::next(MarkerIt));
}

auto BasicBlock::beginDebugProgramRange(iterator FromHere) -> DIIterator {
  if (FromHere == begin())
    return DbgProgramList.begin();
  if (FromHere == end()) {
    // Hmmm, so everything up to the terminator and past it, eh?
    return DbgProgramList.end();
  }
  auto Previous = std::prev(FromHere);
  // If there is no DbgMarker, we are not inhaled, so DbgProgramList must be empty.
  if (!Previous->DbgMarker) {
    assert(DbgProgramList.empty());
    return DbgProgramList.end();
  }
  return std::next(Previous->DbgMarker->getIterator());
}

auto BasicBlock::endDebugProgramRange(iterator FromHere) -> DIIterator {
  if (FromHere == end())
    return DbgProgramList.end();
  // If there is no DbgMarker, we are not inhaled, so DbgProgramList must be empty.
  if (!FromHere->DbgMarker) {
    assert(DbgProgramList.empty());
    return DbgProgramList.end();
  }
  return std::next(FromHere->DbgMarker->getIterator());
}

auto BasicBlock::beginDebugValueRange(iterator FromHere) -> DIIterator {
  return beginDebugProgramRange(FromHere);
}
auto BasicBlock::endDebugValueRange(iterator FromHere) -> DIIterator {
  return getDbgValueMarkerPos(FromHere);
}

auto BasicBlock::getDbgValueMarkerPos(iterator FromHere) -> DIIterator {
  if (FromHere == end())
    return DbgProgramList.end();
  // If there is no DbgMarker, we are not inhaled, so DbgProgramList must be empty.
  if (!FromHere->DbgMarker) {
    assert(DbgProgramList.empty());
    return DbgProgramList.end();
  }
  return FromHere->DbgMarker->getIterator();
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

  DIIterator InsertPoint = InsertAtHead ? beginDebugProgramRange(Dest) : getDbgValueMarkerPos(Dest);

  // Lots of horrible special casing for empty transfers: the dbg.values between
  // two positions could be spliced in dbg.value mode.
  if (First == Last) {
    if (!Src->empty() || DbgProgramList.empty()) {
      // Non-empty block. Are we copying from the head?
      if (!Src->empty() && First == Src->begin() && ReadFromHead) {
        // We need to copy everything from the head of the block over to the destination then.
        DIIterator MoveRangeBegin = Src->DbgProgramList.begin();
        DIIterator MoveRangeEnd = Src->getDbgValueMarkerPos(First);
        for (auto &Item : make_range(MoveRangeBegin, MoveRangeEnd))
          Item.setParent(this);
        DbgProgramList.splice(InsertPoint, Src->DbgProgramList, MoveRangeBegin, MoveRangeEnd);
        return;
      }
    } else {
      // Oh dear; dbg.values in an empty block. Move them over too.
      for (auto &Item : Src->DbgProgramList)
        Item.setParent(this);
      DbgProgramList.splice(InsertPoint, Src->DbgProgramList, Src->DbgProgramList.begin(), Src->DbgProgramList.end());
      return;
    }
  }

  assert(Src->IsInhaled == IsInhaled);

  // Find the start of the portion to copy; if ReadFromHead is specified, this includes First's debug value range, otherwise this is just First's marker.
  DIIterator MoveRangeBegin = ReadFromHead ? Src->beginDebugProgramRange(First) : Src->getDbgValueMarkerPos(First);
  // Find the end of the portion to copy.
  DIIterator MoveRangeEnd = ReadFromTail ? Src->endDebugValueRange(Last) : Src->beginDebugValueRange(Last);

  // Move debug stuff...
  for (auto &Item : make_range(MoveRangeBegin, MoveRangeEnd))
    Item.setParent(this);
  DbgProgramList.splice(InsertPoint, Src->DbgProgramList, MoveRangeBegin, MoveRangeEnd);

  // And move the instructions.
  getInstList().splice(Dest, Src->getInstList(), First, Last);

  flushTerminatorDbgValues();
}

void BasicBlock::blockSplice(iterator Dest, BasicBlock *Src) {
  blockSplice(Dest, Src, Src->begin(), Src->end());
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
