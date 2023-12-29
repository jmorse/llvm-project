//======-- DebugProgramInstruction.cpp - Implement DPValues/DPMarkers --======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/Signals.h"

namespace llvm {

DPValue::DPValue(const DbgVariableIntrinsic *DVI)
    : DebugValueUser(nullptr), Variable(DVI->getVariable()),
      Expression(DVI->getExpression()), DbgLoc(DVI->getDebugLoc()) {
  handleChangedLocation(DVI->getRawLocation());
  switch (DVI->getIntrinsicID()) {
  case Intrinsic::dbg_value:
    Type = LocationType::Value;
    break;
  case Intrinsic::dbg_declare:
    Type = LocationType::Declare;
    break;
  default:
    llvm_unreachable(
        "Trying to create a DPValue with an invalid intrinsic type!");
  }
  isInline = false;
}

DPValue::DPValue(const DPValue &DPV)
    : DebugValueUser(nullptr),
      Variable(DPV.getVariable()), Expression(DPV.getExpression()),
      DbgLoc(DPV.getDebugLoc()), Type(DPV.getType()), ConstantKind(DPV.ConstantKind),
      constant_u(DPV.constant_u) {
//dbgs() << "Src DPValueCreation COPYCONS of " << this << " with location " << DPV.getRawLocation() << "\n";
  if (DPV.getRawLocation())
    setRawLocation(DPV.getRawLocation());
  isInline = false; // Let caller set it if it's important.
}

DPValue::DPValue(Metadata *Location, DILocalVariable *DV, DIExpression *Expr,
                 const DILocation *DI, LocationType Type)
    : DebugValueUser(nullptr), Variable(DV), Expression(Expr), DbgLoc(DI),
      Type(Type) {
//dbgs() << "Src DPValueCreation of " << this << " with location " << Location << "\n";
  setRawLocation(Location);
  isInline = false;
}

void DPValue::deleteInstr() {
//dbgs() << "Delete-instr of DPV " << this << "\n";
//llvm::sys::PrintStackTrace(dbgs(), 0);
  if (!isInline)
    delete this;
  else
    this->~DPValue(); // oh my.
}

iterator_range<DPValue::location_op_iterator> DPValue::location_ops() const {
  auto *MD = getRawLocation();
  // If a Value has been deleted, the "location" for this DPValue will be
  // replaced by nullptr. Return an empty range.
  if (!MD)
    return {location_op_iterator(static_cast<ValueAsMetadata *>(nullptr)),
            location_op_iterator(static_cast<ValueAsMetadata *>(nullptr))};

  // If operand is ValueAsMetadata, return a range over just that operand.
  if (auto *VAM = dyn_cast<ValueAsMetadata>(MD))
    return {location_op_iterator(VAM), location_op_iterator(VAM + 1)};

  // If operand is DIArgList, return a range over its args.
  if (auto *AL = dyn_cast<DIArgList>(MD))
    return {location_op_iterator(AL->args_begin()),
            location_op_iterator(AL->args_end())};

  // Operand is an empty metadata tuple, so return empty iterator.
  assert(cast<MDNode>(MD)->getNumOperands() == 0);
  return {location_op_iterator(static_cast<ValueAsMetadata *>(nullptr)),
          location_op_iterator(static_cast<ValueAsMetadata *>(nullptr))};
}

unsigned DPValue::getNumVariableLocationOps() const {
  if (hasArgList())
    return cast<DIArgList>(getRawLocation())->getArgs().size();
  return 1;
}

Value *DPValue::getVariableLocationOp(unsigned OpIdx) const {
  auto *MD = getRawLocation();
  if (!MD)
    return nullptr;

  if (auto *AL = dyn_cast<DIArgList>(MD))
    return AL->getArgs()[OpIdx]->getValue();
  if (isa<MDNode>(MD))
    return nullptr;
  assert(isa<ValueAsMetadata>(MD) &&
         "Attempted to get location operand from DPValue with none.");
  auto *V = cast<ValueAsMetadata>(MD);
  assert(OpIdx == 0 && "Operand Index must be 0 for a debug intrinsic with a "
                       "single location operand.");
  return V->getValue();
}

static ValueAsMetadata *getAsMetadata(Value *V) {
  return isa<MetadataAsValue>(V) ? dyn_cast<ValueAsMetadata>(
                                       cast<MetadataAsValue>(V)->getMetadata())
                                 : ValueAsMetadata::get(V);
}

void DPValue::replaceVariableLocationOp(Value *OldValue, Value *NewValue,
                                        bool AllowEmpty) {
  assert(NewValue && "Values must be non-null");
//dbgs() << "Src replace " << OldValue << " with  " << NewValue << "\n";
//OldValue->dump();
//NewValue->dump();
//dbgs() << "Cur location is " << getRawLocation() << " for dpv " << this << "\n";
  // Replacing a constant-valued part of this would be ill-conceived.
  assert(!isa<Constant>(OldValue));
  if (ConstantKind != constantKind::None)
    // This doesn't have anything replaceable.
    return;

  auto Locations = location_ops();
  auto OldIt = find(Locations, OldValue);
  if (OldIt == Locations.end()) {
    if (AllowEmpty)
      return;
    llvm_unreachable("OldValue must be a current location");
  }

  if (!hasArgList()) {
    // Set our location to be the MAV wrapping the new Value.
    setRawLocation(isa<MetadataAsValue>(NewValue)
                       ? cast<MetadataAsValue>(NewValue)->getMetadata()
                       : ValueAsMetadata::get(NewValue));
    return;
  }

  // We must be referring to a DIArgList, produce a new operands vector with the
  // old value replaced, generate a new DIArgList and set it as our location.
  SmallVector<ValueAsMetadata *, 4> MDs;
  ValueAsMetadata *NewOperand = getAsMetadata(NewValue);
  for (auto *VMD : Locations)
    MDs.push_back(VMD == *OldIt ? NewOperand : getAsMetadata(VMD));
  setRawLocation(DIArgList::get(getVariableLocationOp(0)->getContext(), MDs));
}

void DPValue::replaceVariableLocationOp(unsigned OpIdx, Value *NewValue) {
  assert(OpIdx < getNumVariableLocationOps() && "Invalid Operand Index");

//dbgs() << "Src direct set of op idx " << OpIdx << " with  " << NewValue << "\n";
//NewValue->dump();
//dbgs() << "  Cur location is " << getRawLocation() << " for dpv " << this << "\n";

  if (!hasArgList()) {
    setRawLocation(isa<MetadataAsValue>(NewValue)
                       ? cast<MetadataAsValue>(NewValue)->getMetadata()
                       : ValueAsMetadata::get(NewValue));
    return;
  }

  SmallVector<ValueAsMetadata *, 4> MDs;
  ValueAsMetadata *NewOperand = getAsMetadata(NewValue);
  for (unsigned Idx = 0; Idx < getNumVariableLocationOps(); ++Idx)
    MDs.push_back(Idx == OpIdx ? NewOperand
                               : getAsMetadata(getVariableLocationOp(Idx)));

  setRawLocation(DIArgList::get(getVariableLocationOp(0)->getContext(), MDs));
}

void DPValue::addVariableLocationOps(ArrayRef<Value *> NewValues,
                                     DIExpression *NewExpr) {
//dbgs() << "LOL ADD VARIABLE LOCATION OPS\n";
  assert(NewExpr->hasAllLocationOps(getNumVariableLocationOps() +
                                    NewValues.size()) &&
         "NewExpr for debug variable intrinsic does not reference every "
         "location operand.");
  assert(!is_contained(NewValues, nullptr) && "New values must be non-null");
  setExpression(NewExpr);
  SmallVector<ValueAsMetadata *, 4> MDs;
  for (auto *VMD : location_ops())
    MDs.push_back(getAsMetadata(VMD));
  for (auto *VMD : NewValues)
    MDs.push_back(getAsMetadata(VMD));
  setRawLocation(DIArgList::get(getVariableLocationOp(0)->getContext(), MDs));
}

void DPValue::setKillLocation() {
  // TODO: When/if we remove duplicate values from DIArgLists, we don't need
  // this set anymore.
//  dbgs() << "Src reset debug value for kill location, i am " << this << "\n";
  resetDebugValue();
#if 0
  SmallPtrSet<Value *, 4> RemovedValues;
  for (Value *OldValue : location_ops()) {
    if (!RemovedValues.insert(OldValue).second)
      continue;
    Value *Poison = PoisonValue::get(OldValue->getType());
    replaceVariableLocationOp(OldValue, Poison);
  }
#endif
}

bool DPValue::isKillLocation() const {
  bool res = (getNumVariableLocationOps() == 0 &&
          !getExpression()->isComplex()) ||
         any_of(location_ops(), [](Value *V) { return isa<UndefValue>(V); });
  if (res)
    return true;

  // One location that's empty,
  if (getNumVariableLocationOps() == 1 && !getRawLocation())
    return true;
  return false;
}

std::optional<uint64_t> DPValue::getFragmentSizeInBits() const {
  if (auto Fragment = getExpression()->getFragmentInfo())
    return Fragment->SizeInBits;
  return getVariable()->getSizeInBits();
}

DPValue *DPValue::clone() const { return new DPValue(*this); }

DbgVariableIntrinsic *
DPValue::createDebugIntrinsic(Module *M, Instruction *InsertBefore) const {
  [[maybe_unused]] DICompileUnit *Unit =
      getDebugLoc().get()->getScope()->getSubprogram()->getUnit();
  assert(M && Unit &&
         "Cannot clone from BasicBlock that is not part of a Module or "
         "DICompileUnit!");
  LLVMContext &Context = getDebugLoc()->getContext();
  Value *Args[] = {MetadataAsValue::get(Context, getRawLocation()),
                   MetadataAsValue::get(Context, getVariable()),
                   MetadataAsValue::get(Context, getExpression())};
  Function *IntrinsicFn;

  // Work out what sort of intrinsic we're going to produce.
  switch (getType()) {
  case DPValue::LocationType::Declare:
    IntrinsicFn = Intrinsic::getDeclaration(M, Intrinsic::dbg_declare);
    break;
  case DPValue::LocationType::Value:
    IntrinsicFn = Intrinsic::getDeclaration(M, Intrinsic::dbg_value);
    break;
  case DPValue::LocationType::End:
  case DPValue::LocationType::Any:
    llvm_unreachable("Invalid LocationType");
    break;
  }

  // Create the intrinsic from this DPValue's information, optionally insert
  // into the target location.
  DbgVariableIntrinsic *DVI = cast<DbgVariableIntrinsic>(
      CallInst::Create(IntrinsicFn->getFunctionType(), IntrinsicFn, Args));
  DVI->setTailCall();
  DVI->setDebugLoc(getDebugLoc());
  if (InsertBefore)
    DVI->insertBefore(InsertBefore);

  return DVI;
}

void DPValue::handleChangedLocation(Metadata *NewLocation) {
  // Test for various constant and undef things.
//  dbgs() << "  Reset debug value for changing to " << NewLocation << " from " << getRawLocation() << ", I am " << this << "\n";
  resetDebugValue();
  if (!NewLocation)
    return;
  if (ValueAsMetadata *VAM = dyn_cast<ValueAsMetadata>(NewLocation)) {
    Value *V = VAM->getValue();
    if (isa<UndefValue>(V)) {
//dbgs() << "  AKERHSERLY set to null\n";
      resetDebugValue();
      return;
    } else if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      if (cast<IntegerType>(CI->getType())->getBitWidth() <= 64) {
        if (CI->isNegative()) {
          ConstantKind = Signed;
          constant_u.sl = CI->getSExtValue();
        } else {
          ConstantKind = Unsigned;
          constant_u.ul = CI->getZExtValue();
        }
//dbgs() << "  AKERHSERLY set to null\n";
        return;
      }
    } else if (ConstantFP *FP = dyn_cast<ConstantFP>(V)) {
      llvm::Type *TT = FP->getType();
      if (TT->isFloatTy()) {
        ConstantKind = Float;
        constant_u.f = FP->getValue().convertToFloat();
      } else {
        if (!TT->isDoubleTy()) {
		// Yeah, this can happen when we get x86_80 floats
		// Just bung a number in there, we need to make DPValue
		// store an APFloat in this case anyway.
		ConstantKind = Double;
		constant_u.d = 1234.0;
	} else {
          ConstantKind = Double;
          constant_u.d = FP->getValue().convertToDouble();
	}
      }
//dbgs() << "  AKERHSERLY set to null\n";
      return;
    } else if (isa<ConstantPointerNull>(V)) {
      ConstantKind = Nullptr;
//dbgs() << "  AKERHSERLY set to null\n";
      return;
    }
  }

//dbgs() << "  committed\n";
  resetDebugValue(NewLocation);
}

const BasicBlock *DPValue::getParent() const {
  return Marker->MarkedInstr->getParent();
}

BasicBlock *DPValue::getParent() { return Marker->MarkedInstr->getParent(); }

BasicBlock *DPValue::getBlock() { return Marker->getParent(); }

const BasicBlock *DPValue::getBlock() const { return Marker->getParent(); }

Function *DPValue::getFunction() { return getBlock()->getParent(); }

const Function *DPValue::getFunction() const { return getBlock()->getParent(); }

Module *DPValue::getModule() { return getFunction()->getParent(); }

const Module *DPValue::getModule() const { return getFunction()->getParent(); }

LLVMContext &DPValue::getContext() { return getBlock()->getContext(); }

const LLVMContext &DPValue::getContext() const {
  return getBlock()->getContext();
}

///////////////////////////////////////////////////////////////////////////////

// An empty, global, DPMarker for the purpose of describing empty ranges of
// DPValues.
DPMarker DPMarker::EmptyDPMarker;

void DPMarker::dropDPValues() {
  while (!StoredDPValues.empty()) {
    auto It = StoredDPValues.begin();
    DPValue *DPV = &*It;
    StoredDPValues.erase(It);
    DPV->deleteInstr();
  }
}

void DPMarker::dropOneDPValue(DPValue *DPV) {
  assert(DPV->getMarker() == this);
  StoredDPValues.erase(DPV->getIterator());
  DPV->deleteInstr();
}

const BasicBlock *DPMarker::getParent() const {
  return MarkedInstr->getParent();
}

BasicBlock *DPMarker::getParent() { return MarkedInstr->getParent(); }

void DPMarker::removeMarker() {
#if 0
dbgs() << "Removing marker from ";
MarkedInstr->dump();
dbgs() << "Before";
dump();
#endif
  // Are there any DPValues in this DPMarker? If not, nothing to preserve.
  Instruction *Owner = MarkedInstr;
  if (StoredDPValues.empty()) {
    eraseFromParent();
    Owner->DbgMarker = nullptr;
    return;
  }

  // The attached DPValues need to be preserved; attach them to the next
  // instruction. If there isn't a next instruction, put them on the
  // "trailing" list.
  DPMarker *NextMarker = Owner->getParent()->getNextMarker(Owner);
  if (NextMarker) {
#if 0
dbgs() << "Next";
NextMarker->dump();
#endif
    NextMarker->absorbDebugValues(*this, true);
    // that erases us!
  } else {
    // We can avoid a deallocation -- just store this marker onto the next
    // instruction. Are we at the end of the block?
    BasicBlock::iterator NextIt = std::next(Owner->getIterator());
    if (NextIt == getParent()->end()) {
      getParent()->setTrailingDPValues(this);
      MarkedInstr = nullptr;
    } else {
      NextIt->DbgMarker = this;
      MarkedInstr = &*NextIt;
    }
    Owner->DbgMarker = nullptr;
  }
assert(!Owner->DbgMarker);
//dbgs() << "After";
//dump();
}

void DPMarker::removeFromParent() {
if (MarkedInstr) // might be a dangler?
  MarkedInstr->DbgMarker = nullptr;
  MarkedInstr = nullptr;
}

void DPMarker::eraseFromParent() {
  if (MarkedInstr)
    removeFromParent();
  dropDPValues();

  // Delete down the chain -- nothing further down should either mark anything
  // or have any DPValues in a list.
  MarkedInstr = nullptr;
  DPMarker *ToFree = this;
  while (ToFree) {
    assert(ToFree->MarkedInstr == nullptr);
    assert(ToFree->StoredDPValues.empty());
    DPMarker *tmp = ToFree;
    ToFree = ToFree->ToFreeChain;
    delete tmp;
  }
}

iterator_range<DPValue::self_iterator> DPMarker::getDbgValueRange() {
  return make_range(StoredDPValues.begin(), StoredDPValues.end());
}

void DPValue::removeFromParent() {
  getMarker()->StoredDPValues.erase(getIterator());
}

DPValue *DPValue::unlinkFromParent() {
  getMarker()->StoredDPValues.erase(getIterator());

  if (isInline) {
    DPValue *Clone = clone();
    deleteInstr(); // delete ourselves!
    return Clone;
  } else {
    return this;
  }
}

void DPValue::eraseFromParent() {
  removeFromParent();
  deleteInstr();
}

bool DPValue::isConstant() {
  return ConstantKind != constantKind::None;
}

Value *DPValue::coughUpConstant() {
  LLVMContext &Ctx = getParent()->getContext();
  switch(ConstantKind) {
  default:
    llvm_unreachable("wrong constant kind DPValue?");
  case constantKind::Unsigned: {
    llvm::Type *TT = IntegerType::get(Ctx, 64);
    return ConstantInt::get(TT, constant_u.ul, false);
  }
  case constantKind::Signed: {
    llvm::Type *TT = IntegerType::get(Ctx, 64);
    return ConstantInt::get(TT, constant_u.sl, true);
  }
  case constantKind::Float: {
    llvm::Type *TT = Type::getFloatTy(Ctx);
    return ConstantFP::get(TT, APFloat(constant_u.f));
  }
  case constantKind::Double: {
    llvm::Type *TT = Type::getDoubleTy(Ctx);
    return ConstantFP::get(TT, APFloat(constant_u.d));
  }
  case constantKind::Nullptr: {
    llvm::PointerType *TT = PointerType::get(Ctx, 0);
    return ConstantPointerNull::get(TT);
  }
  }
}

void DPMarker::insertDPValue(DPValue *New, bool InsertAtHead) {
  auto It = InsertAtHead ? StoredDPValues.begin() : StoredDPValues.end();
  StoredDPValues.insert(It, *New);
  New->setMarker(this);
}

void DPMarker::absorbDebugValues(DPMarker &Src, bool InsertAtHead) {
  auto It = InsertAtHead ? StoredDPValues.begin() : StoredDPValues.end();
  bool HasInline = false;
  for (DPValue &DPV : make_early_inc_range(Src.StoredDPValues)) {
    DPV.setMarker(this);
    HasInline |= DPV.isInline;
  }

  StoredDPValues.splice(It, Src.StoredDPValues);

  if (HasInline) {
    Src.removeFromParent();
    // What if both of these things have to-free chains?
    // Seek out the end of ours and put Src there.
    DPMarker *EndOfList = this;
    while (EndOfList->ToFreeChain)
      EndOfList = EndOfList->ToFreeChain;
    EndOfList->ToFreeChain = &Src;
  } else {
    Src.eraseFromParent();
  }
// XXX -- this effectively invalidates the Src, need to check all callsites.
}

void DPMarker::absorbDebugValues(iterator_range<DPValue::self_iterator> Range,
                                 DPMarker &Src, bool InsertAtHead) {
// Right -- actually this function is only ever called during CGP transaction
// rollback, and what it does is peel apart the DPValues in Range from where
// they used to be. We can just move them; as normal, but we need to move the
// pointer ownership between DPMarkers so that they're freed at the right time.
// The hazard here is that we end up with overlapping DPValue ranges in the
// two markers owned by overlapping sets of DPMarkers; that should never happen
// because we have one call-site.

// Collect the set of owning DPMarkers.
DPMarker *Owner = &Src;
SmallPtrSet<DPMarker *, 8> Owners, MovingSet;
while (Owner) {
	if (Owner->NumInline)
    Owners.insert(Owner);
  Owner = Owner->ToFreeChain;
}

  for (DPValue &DPV : Range) {
    if (!DPV.isInline) {
      DPV.setMarker(this);
    } else {
      // Work out which marker this is and move it from one place to another.
      intptr_t dpvaddr = reinterpret_cast<intptr_t>(&DPV);
      bool found = false;
      for (DPMarker *DPM : Owners) {
        intptr_t startaddr = reinterpret_cast<intptr_t>(DPM);
        intptr_t end = reinterpret_cast<intptr_t>(DPM->getInline(DPM->NumInline-1));
        if (dpvaddr >= startaddr && dpvaddr <= end) {
          MovingSet.insert(DPM);
          found = true;
          break;
        }
      }
      assert(found && "Didn't find an owning DPM for inline thingy?");
      DPV.setMarker(this);
    }
  }
  auto InsertPos =
      (InsertAtHead) ? StoredDPValues.begin() : StoredDPValues.end();

  StoredDPValues.splice(InsertPos, Src.StoredDPValues, Range.begin(),
                        Range.end());

  // Produce two distinct sets of Markers to be owned,
  for (auto *DPM : MovingSet)
    Owners.erase(DPM);

  // Replace Src owned markers with the new owner set,
  Owner = &Src;
  while (Owner != nullptr) {
    if (MovingSet.count(Owner->ToFreeChain)) {
      Owner->ToFreeChain = Owner->ToFreeChain->ToFreeChain;
    } else {
      Owner = Owner->ToFreeChain;
    }
  }

  // Add the newly owned markers to this' owned list.
  for (DPMarker *DPM : MovingSet) {
    DPM->ToFreeChain = ToFreeChain;
    ToFreeChain = DPM;
  }
}

iterator_range<simple_ilist<DPValue>::iterator> DPMarker::cloneDebugInfoFrom(
    DPMarker *From, std::optional<simple_ilist<DPValue>::iterator> from_here,
    bool InsertAtHead) {
  DPValue *First = nullptr;
  // Work out what range of DPValues to clone: normally all the contents of the
  // "From" marker, optionally we can start from the from_here position down to
  // end().
  auto Range =
      make_range(From->StoredDPValues.begin(), From->StoredDPValues.end());
  if (from_here.has_value())
    Range = make_range(*from_here, From->StoredDPValues.end());

  // Clone each DPValue and insert into StoreDPValues; optionally place them at
  // the start or the end of the list.
  auto Pos = (InsertAtHead) ? StoredDPValues.begin() : StoredDPValues.end();
  for (DPValue &DPV : Range) {
    DPValue *New = DPV.clone();
    New->setMarker(this);
    StoredDPValues.insert(Pos, *New);
    if (!First)
      First = New;
  }

  if (!First)
    return {StoredDPValues.end(), StoredDPValues.end()};

  if (InsertAtHead)
    // If InsertAtHead is set, we cloned a range onto the front of of the
    // StoredDPValues collection, return that range.
    return {StoredDPValues.begin(), Pos};
  else
    // We inserted a block at the end, return that range.
    return {First->getIterator(), StoredDPValues.end()};
}

DPValue *
DPMarker::getInline(unsigned int Idx) {
  assert(Idx < NumInline);
  uintptr_t Ptr = reinterpret_cast<uintptr_t>(this);
  Ptr += sizeof(DPMarker);
  Ptr += (Idx * sizeof(DPValue));
  return reinterpret_cast<DPValue *>(Ptr);
}

DPMarker::DPMarker(unsigned int NumInline) {
  this->NumInline = NumInline;
}

DPMarker *
DPMarker::allocWithInline(unsigned int Num) {
  unsigned int size_wanted = sizeof(DPMarker) + (sizeof(DPValue) * Num);
  DPMarker *DPM = static_cast<DPMarker*>(::operator new(size_wanted));
  // In-place constructor call.
  new (DPM) DPMarker(Num);
  return DPM;
}

DPMarker *
DPMarker::cloneDebugInfoFromInline(Instruction *inst_from, DPMarker *From,
                       std::optional<simple_ilist<DPValue>::iterator> from_here) {
  // Form the source range,
  auto Range =
      make_range(From->StoredDPValues.begin(), From->StoredDPValues.end());
  if (from_here.has_value())
    Range = make_range(*from_here, From->StoredDPValues.end());

  unsigned NumElems = std::distance(Range.begin(), Range.end());
  DPMarker *DPM = allocWithInline(NumElems);
  // XXX -- what do to when there are none, be an error?

  // Iterate through the range, constructing in-place in the allocated blobs,
  // and setting the "inline" flag.
  unsigned int Count = 0 ;
  for (DPValue &DPV : Range) {
    DPValue *NewMem = DPM->getInline(Count);
    new (NewMem) DPValue(DPV);
    NewMem->setMarker(DPM);
    NewMem->isInline = true;
    DPM->StoredDPValues.insert(DPM->StoredDPValues.end(), *NewMem);
    ++Count;
  }

  return DPM;
}


} // end namespace llvm

