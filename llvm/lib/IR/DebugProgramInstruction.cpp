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

DebugProgramInstruction::DebugProgramInstruction(DPInstrType InstrType)
  : Parent(nullptr), InstrType(InstrType) {
}

void DebugProgramInstruction::deleteInstr() {
  switch (getInstrType())
  {
  case DebugProgramInstruction::DPInstrType::Marker:
    delete static_cast<DPMarker*>(this);
    break;
  case DebugProgramInstruction::DPInstrType::Value:
    delete static_cast<DPValue*>(this);
    break;
  default:
    llvm_unreachable("Invalid DebugProgramInstruction type");
  }
}

const Module *DebugProgramInstruction::getModule() const {
  return getFunction()->getParent();
}
const Function *DebugProgramInstruction::getFunction() const {
  return getParent()->getParent();
}
LLVMContext &DebugProgramInstruction::getContext() const { return Parent->getContext(); }

void DebugProgramInstruction::removeFromParent() {
  getParent()->DbgProgramList.erase(getIterator());
}
void DebugProgramInstruction::eraseFromParent() {
  removeFromParent();
  deleteInstr();
}

//===----------------------------------------------------------------------===//
/// DPValue - This is the common base class for debug info
/// intrinsics for variables.
///

DPValue::DPValue(const DbgVariableIntrinsic *DVI) :
  DebugProgramInstruction(DPInstrType::Value), DebugValueUser(DVI->getRawLocation()),
  Variable(DVI->getVariable()), Expression(DVI->getExpression()),
  DbgLoc(DVI->getDebugLoc()) {
  setParent(const_cast<BasicBlock *>(DVI->getParent()));
  switch (DVI->getIntrinsicID()) {
    case Intrinsic::dbg_value: {
      Type = LocationType::Value;
      break;
    }
    case Intrinsic::dbg_addr: {
      Type = LocationType::Address;
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
  DebugProgramInstruction(DPInstrType::Value), DebugValueUser(DPV.getRawLocation()),
  Type(DPV.getType()), Variable(DPV.getVariable()),
  Expression(DPV.getExpression()), DbgLoc(DPV.getDebugLoc()) {}


iterator_range<DPValue::location_op_iterator>
DPValue::location_ops() const {
  auto *MD = getRawLocation();
  assert(MD && "Location of DPValue should be non-null.");

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

Optional<uint64_t> DPValue::getFragmentSizeInBits() const {
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
    case DPValue::LocationType::Address: {
      IntrinsicFn = Intrinsic::getDeclaration(M, Intrinsic::dbg_addr);
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
    DVI->insertDebugBefore(InsertBefore);
  return DVI;
}

void DPValue::handleChangedLocation(Metadata *NewLocation) {
  resetDebugValue(NewLocation);
}

} // end namespace llvm
