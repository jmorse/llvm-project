//===-- DebugProgramInstruction.cpp - Implement the DebugProgramInstruction class --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the DebugProgramInstruction class for the IR library.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/IntrinsicInst.h"

namespace llvm {

//===----------------------------------------------------------------------===//
/// DPValue - This is the common base class for debug info
/// intrinsics for variables.
///

DPValue::DPValue(const DbgVariableIntrinsic *DVI) :
  DebugValueUser(DVI->getRawLocation()),
  Variable(DVI->getVariable()), Expression(DVI->getExpression()),
  DbgLoc(DVI->getDebugLoc()) {
  switch (DVI->getIntrinsicID()) {
    case Intrinsic::dbg_value: {
      Type = LocationType::Value;
      break;
    }
    case Intrinsic::dbg_declare: {
      Type = LocationType::Declare;
      break;
    }
    default: llvm_unreachable("Trying to create a DPValue with an invalid intrinsic type!");
  }
}

DPValue::DPValue(const DPValue &DPV) :
  DebugValueUser(DPV.getRawLocation()),
  Type(DPV.getType()), Variable(DPV.getVariable()),
  Expression(DPV.getExpression()), DbgLoc(DPV.getDebugLoc()) {}

DPValue::DPValue(Metadata *Location, DILocalVariable *DV, DIExpression *Expr, const DILocation *DI)
  : DebugValueUser(Location), Variable(DV), Expression(Expr), DbgLoc(DI) {
  Type = LocationType::Value;
}

void DPValue::deleteInstr() {
  delete this;
}

iterator_range<DPValue::location_op_iterator>
DPValue::location_ops() const {
  auto *MD = getRawLocation();
//  assert(MD && "Location of DPValue should be non-null.");
  if (!MD)
    return {location_op_iterator(static_cast<ValueAsMetadata *>(nullptr)),
            location_op_iterator(static_cast<ValueAsMetadata *>(nullptr))};

  // If operand is ValueAsMetadata, return a range over just that operand.
  if (auto *VAM = dyn_cast<ValueAsMetadata>(MD)) {
    return {location_op_iterator(VAM), location_op_iterator(VAM + 1)};
  }
  // If operand is DIArgList, return a range over its args.
  if (auto *AL = dyn_cast<DIArgList>(MD))
    return {location_op_iterator(AL->args_begin()),
            location_op_iterator(AL->args_end())};
  // Operand must be an empty metadata tuple, so return empty iterator.
  return {location_op_iterator(static_cast<ValueAsMetadata *>(nullptr)),
          location_op_iterator(static_cast<ValueAsMetadata *>(nullptr))};
}

Value *DPValue::getVariableLocationOp(unsigned OpIdx) const {
  auto *MD = getRawLocation();
  if (!MD)
    return nullptr;

  if (auto *AL = dyn_cast<DIArgList>(MD))
    return AL->getArgs()[OpIdx]->getValue();
  if (isa<MDNode>(MD))
    return nullptr;
  assert(
      isa<ValueAsMetadata>(MD) &&
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

void DPValue::replaceVariableLocationOp(Value *OldValue,
                                        Value *NewValue, bool AllowEmpty) {
  assert(NewValue && "Values must be non-null");
  auto Locations = location_ops();
  auto OldIt = find(Locations, OldValue);
  if (OldIt == Locations.end()) {
    if (AllowEmpty)
      return;
    llvm_unreachable("OldValue must be a current location");
  }

  if (!hasArgList()) {
    setRawLocation(isa<MetadataAsValue>(NewValue)
                     ? cast<MetadataAsValue>(NewValue)->getMetadata()
                     : ValueAsMetadata::get(NewValue));
    return;
  }
  SmallVector<ValueAsMetadata *, 4> MDs;
  ValueAsMetadata *NewOperand = getAsMetadata(NewValue);
  for (auto *VMD : Locations)
    MDs.push_back(VMD == *OldIt ? NewOperand : getAsMetadata(VMD));
  setRawLocation(DIArgList::get(getVariableLocationOp(0)->getContext(), MDs));
}
void DPValue::replaceVariableLocationOp(unsigned OpIdx,
                                        Value *NewValue) {
  assert(OpIdx < getNumVariableLocationOps() && "Invalid Operand Index");
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

std::optional<uint64_t> DPValue::getFragmentSizeInBits() const {
  if (auto Fragment = getExpression()->getFragmentInfo())
    return Fragment->SizeInBits;
  return getVariable()->getSizeInBits();
}

DPValue *DPValue::clone() const {
  return new DPValue(*this);
}

DbgVariableIntrinsic *DPValue::createDebugIntrinsic(Module *M, Instruction *InsertBefore) const {
  DICompileUnit *Unit = getDebugLoc().get()->getScope()->getSubprogram()->getUnit();
  assert(M && Unit && "Cannot clone from BasicBlock that is not part of a Module or DICompileUnit!");
  LLVMContext &Context = getDebugLoc()->getContext();
  Value *Args[] = {MetadataAsValue::get(Context, getRawLocation()),
                   MetadataAsValue::get(Context, getVariable()),
                   MetadataAsValue::get(Context, getExpression())};
  Function *IntrinsicFn;
  switch(getType()) {
    case DPValue::LocationType::Declare: {
      IntrinsicFn = Intrinsic::getDeclaration(M, Intrinsic::dbg_declare);
      break;
    }
    case DPValue::LocationType::Value: {
      IntrinsicFn = Intrinsic::getDeclaration(M, Intrinsic::dbg_value);
      break;
    }
    default:
      llvm_unreachable("Invalid LocationType for DPValue!");
  }
  DbgVariableIntrinsic *DVI = cast<DbgVariableIntrinsic>(CallInst::Create(IntrinsicFn->getFunctionType(), IntrinsicFn, Args));
  DVI->setDebugLoc(getDebugLoc());
  if (InsertBefore)
    DVI->insertBefore(InsertBefore);
  return DVI;
}

void DPValue::handleChangedLocation(Metadata *NewLocation) {
  resetDebugValue(NewLocation);
}

const BasicBlock *DPValue::getParent() const {
  return Marker->MarkedInstr->getParent();
}

BasicBlock *DPValue::getParent() {
  return Marker->MarkedInstr->getParent();
}

BasicBlock *DPValue::getBlock() {
  return Marker->getParent();
}

const BasicBlock *DPValue::getBlock() const {
  return Marker->getParent();
}

Function *DPValue::getFunction() {
  return getBlock()->getParent();
}

const Function *DPValue::getFunction() const {
  return getBlock()->getParent();
}

Module *DPValue::getModule() {
  return getFunction()->getParent();
}

const Module *DPValue::getModule() const {
  return getFunction()->getParent();
}

LLVMContext &DPValue::getContext() {
  return getBlock()->getContext();
}

const LLVMContext &DPValue::getContext() const {
  return getBlock()->getContext();
}


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

BasicBlock *DPMarker::getParent() {
  return MarkedInstr->getParent();
}

void DPMarker::removeMarker() {
  // Are there any dbg.values on the corresponding inst? If not, we're golden.
  Instruction *Owner = MarkedInstr;
  if (StoredDPValues.empty()) {
    eraseFromParent();
    Owner->DbgMarker = nullptr;
    return;
  }

  // Need to update the attached DPValues to point at the next one. LEave as
  // nullptr if we're going to dangle off the end.
  // Get next marker: to be refactored in future patch,
  BasicBlock::iterator NextInst = std::next(Owner->getIterator());
  DPMarker *NextMarker;
  if (NextInst == Owner->getParent()->end())
    NextMarker = &Owner->getParent()->TrailingDPValues;
  else
    NextMarker = NextInst->DbgMarker;
  NextMarker->absorbDebugValues(*this, true);

  eraseFromParent();
  Owner->DbgMarker = nullptr;
  return;
}

void DPMarker::removeFromParent() {
  MarkedInstr->DbgMarker = nullptr;
  MarkedInstr = nullptr;
}

void DPMarker::eraseFromParent() {
  removeFromParent();
  dropDPValues();
  delete this;
}

iterator_range<DPValue::self_iterator> DPMarker::getDbgValueRange() {
  return make_range(StoredDPValues.begin(), StoredDPValues.end());
}

void DPValue::removeFromParent() {
  getMarker()->StoredDPValues.erase(getIterator());
}

void DPValue::eraseFromParent() {
  removeFromParent();
  deleteInstr();
}

void DPMarker::insertDPValue(DPValue *New, bool InsertAtHead) {
  auto It = InsertAtHead ? StoredDPValues.begin() : StoredDPValues.end();
  StoredDPValues.insert(It, *New);
  New->setMarker(this);
}

void DPMarker::absorbDebugValues(DPMarker &Src, bool InsertAtHead) {
  auto It = InsertAtHead ? StoredDPValues.begin() : StoredDPValues.end();
  for (DPValue &DPV : Src.StoredDPValues)
    DPV.setMarker(this);

  StoredDPValues.splice(It, Src.StoredDPValues);
}

iterator_range<simple_ilist<DPValue>::iterator> DPMarker::cloneDebugInfoFrom(DPMarker *From, std::optional<simple_ilist<DPValue>::iterator> from_here) {
  DPValue *First = nullptr;
  auto Range = make_range(From->StoredDPValues.begin(), From->StoredDPValues.end());
  if (from_here.has_value())
    Range = make_range(*from_here, From->StoredDPValues.end());

  for (DPValue &DPV : Range) {
    DPValue *New = DPV.clone();
    New->setMarker(this);
    StoredDPValues.insert(StoredDPValues.end(), *New);
    if (!First)
      First = New;
  }
  if (!First)
    return {StoredDPValues.end(), StoredDPValues.end()};

  return First->getIterator();
}

} // end namespace llvm
