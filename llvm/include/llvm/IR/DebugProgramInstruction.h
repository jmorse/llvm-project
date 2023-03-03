//===-- llvm/DebugProgramInstruction.h - Stream of debug info -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the Instruction class, which is the
// base class for all of the LLVM instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DEBUGPROGRAMINSTRUCTION_H
#define LLVM_IR_DEBUGPROGRAMINSTRUCTION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Bitfields.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/ADT/iterator.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/TrackingMDRef.h"
#include "llvm/Support/AtomicOrdering.h"
#include <cstdint>
#include <utility>

namespace llvm {

class BasicBlock;
class DIBuilder;
class FastMathFlags;
class Instruction;
class MDNode;
class Module;
class DbgVariableIntrinsic;
class DebugProgramInstruction;
class raw_ostream;
struct AAMDNodes;

class DebugProgramInstruction : public ilist_node_with_parent<DebugProgramInstruction, BasicBlock> {
public:
  enum class DPInstrType {
    Value,
    Marker
  };
private:
  BasicBlock *Parent;
  DPInstrType InstrType;
  
public:
  DebugProgramInstruction(DPInstrType InstrType);
  DebugProgramInstruction(DebugProgramInstruction& DPI) = delete;

  void deleteInstr();

  DPInstrType getInstrType() { return InstrType; }

  inline const BasicBlock *getParent() const { return Parent; }
  inline       BasicBlock *getParent()       { return Parent; }

  void setParent(BasicBlock *NewParent) {
    Parent = NewParent;
  }

  bool isMarker() const { return InstrType == DPInstrType::Marker; }
  bool isValue() const { return InstrType == DPInstrType::Value; }

  /// Return the module owning the function this DebugProgramInstruction belongs to
  /// or nullptr it the function does not have a module.
  ///
  /// Note: this is undefined behavior if the DebugProgramInstruction does not have a
  /// parent, or the parent basic block does not have a parent function.
  const Module *getModule() const;
  Module *getModule() {
    return const_cast<Module *>(
                           static_cast<const DebugProgramInstruction *>(this)->getModule());
  }
  
  LLVMContext &getContext() const;

  /// Return the function this DebugProgramInstruction belongs to.
  ///
  /// Note: it is undefined behavior to call this on a DebugProgramInstruction not
  /// currently inserted into a function.
  const Function *getFunction() const;
  Function *getFunction() {
    return const_cast<Function *>(
                         static_cast<const DebugProgramInstruction *>(this)->getFunction());
  }

  /// This method unlinks 'this' from the containing basic block, but does not
  /// delete it.
  void removeFromParent();
  void eraseFromParent();
  
  /// Support for debugging, callable in GDB: V->dump()
  void dump() const;

  /// Implement operator<< on DebugProgramInstruction.
  void print(raw_ostream &O, bool IsForDebug = false) const;
  void print(raw_ostream &ROS, ModuleSlotTracker &MST, bool IsForDebug) const;
  
};

inline raw_ostream &operator<<(raw_ostream &OS, const DebugProgramInstruction &DPI) {
  DPI.print(OS);
  return OS;
}

// Make DebugValueUser inheritance private so as to not clog up the interface 
// here.
class DPValue : public DebugProgramInstruction, private DebugValueUser {
friend class DebugValueUser;
public:
  enum class LocationType {
    Declare,
    Address,
    Value,
  };
  // Is either a ValueAsMetadata or a DIArgList. Tracking behaviour is only
  // needed for ValueAsMetadata, 
  LocationType Type;
  TypedTrackingMDRef<DILocalVariable> Variable;
  TypedTrackingMDRef<DIExpression> Expression;
  DebugLoc DbgLoc;

  DPValue(const DbgVariableIntrinsic *DVI);
  DPValue(const DPValue &DPV);

  // Iterator for ValueAsMetadata that internally uses direct pointer iteration
  // over either a ValueAsMetadata* or a ValueAsMetadata**, dereferencing to the
  // ValueAsMetadata .
  class location_op_iterator
      : public iterator_facade_base<location_op_iterator,
                                    std::bidirectional_iterator_tag, Value *> {
    PointerUnion<ValueAsMetadata *, ValueAsMetadata **> I;

  public:
    location_op_iterator(ValueAsMetadata *SingleIter) : I(SingleIter) {}
    location_op_iterator(ValueAsMetadata **MultiIter) : I(MultiIter) {}

    location_op_iterator(const location_op_iterator &R) : I(R.I) {}
    location_op_iterator &operator=(const location_op_iterator &R) {
      I = R.I;
      return *this;
    }
    bool operator==(const location_op_iterator &RHS) const {
      return I == RHS.I;
    }
    const Value *operator*() const {
      ValueAsMetadata *VAM = I.is<ValueAsMetadata *>()
                                 ? I.get<ValueAsMetadata *>()
                                 : *I.get<ValueAsMetadata **>();
      return VAM->getValue();
    };
    Value *operator*() {
      ValueAsMetadata *VAM = I.is<ValueAsMetadata *>()
                                 ? I.get<ValueAsMetadata *>()
                                 : *I.get<ValueAsMetadata **>();
      return VAM->getValue();
    }
    location_op_iterator &operator++() {
      if (I.is<ValueAsMetadata *>())
        I = I.get<ValueAsMetadata *>() + 1;
      else
        I = I.get<ValueAsMetadata **>() + 1;
      return *this;
    }
    location_op_iterator &operator--() {
      if (I.is<ValueAsMetadata *>())
        I = I.get<ValueAsMetadata *>() - 1;
      else
        I = I.get<ValueAsMetadata **>() - 1;
      return *this;
    }
  };

  /// Get the locations corresponding to the variable referenced by the debug
  /// info intrinsic.  Depending on the intrinsic, this could be the
  /// variable's value or its address.
  iterator_range<location_op_iterator> location_ops() const;

  Value *getVariableLocationOp(unsigned OpIdx) const;

  void replaceVariableLocationOp(Value *OldValue, Value *NewValue, bool AllowEmpty = false);
  void replaceVariableLocationOp(unsigned OpIdx, Value *NewValue);
  /// Adding a new location operand will always result in this intrinsic using
  /// an ArgList, and must always be accompanied by a new expression that uses
  /// the new operand.
  void addVariableLocationOps(ArrayRef<Value *> NewValues,
                              DIExpression *NewExpr);

  void setVariable(DILocalVariable *NewVar) {
    Variable.reset(NewVar);
  }

  void setExpression(DIExpression *NewExpr) {
    Expression.reset(NewExpr);
  }

  unsigned getNumVariableLocationOps() const {
    if (hasArgList())
      return cast<DIArgList>(getRawLocation())->getArgs().size();
    return 1;
  }

  bool hasArgList() const { return isa<DIArgList>(getRawLocation()); }
  /// Returns true if this DPValue has no empty MDNodes in its location list.
  bool hasValidLocation() const { return getVariableLocationOp(0) != nullptr; }

  /// Does this describe the address of a local variable. True for dbg.addr
  /// and dbg.declare, but not dbg.value, which describes its value.
  bool isAddressOfVariable() const {
    return Type != LocationType::Value;
  }
  LocationType getType() const {
    return Type;
  }

  DebugLoc getDebugLoc() const { return DbgLoc; }
  void setDebugLoc(DebugLoc Loc) { DbgLoc = std::move(Loc); }

  void setUndef() {
    // TODO: When/if we remove duplicate values from DIArgLists, we don't need
    // this set anymore.
    SmallPtrSet<Value *, 4> RemovedValues;
    for (Value *OldValue : location_ops()) {
      if (!RemovedValues.insert(OldValue).second)
        continue;
      Value *Undef = UndefValue::get(OldValue->getType());
      replaceVariableLocationOp(OldValue, Undef);
    }
  }

  bool isUndef() const {
    return (getNumVariableLocationOps() == 0 &&
            !getExpression()->isComplex()) ||
           any_of(location_ops(), [](Value *V) { return isa<UndefValue>(V); });
  }

  DILocalVariable *getVariable() const {
    return Variable.get();
  }

  DIExpression *getExpression() const {
    return Expression.get();
  }

  Metadata *getRawLocation() const {
    return DebugValue;
  }

  /// Use of this should generally be avoided; instead,
  /// replaceVariableLocationOp and addVariableLocationOps should be used where
  /// possible to avoid creating invalid state.
  void setRawLocation(Metadata *NewLocation) {
    assert(
        (isa<ValueAsMetadata>(NewLocation) || isa<DIArgList>(NewLocation) ||
         isa<MDNode>(NewLocation)) &&
        "Location for a DPValue must be either ValueAsMetadata or DIArgList");
    resetDebugValue(NewLocation);
  }

  /// Get the size (in bits) of the variable, or fragment of the variable that
  /// is described.
  Optional<uint64_t> getFragmentSizeInBits() const;

  DPValue *clone() const;
  DbgVariableIntrinsic *createDebugIntrinsic(Module *M, Instruction *InsertBefore) const;
  void handleChangedLocation(Metadata *NewLocation);

};

class DPMarker : public DebugProgramInstruction {
public:
  DPMarker() : DebugProgramInstruction(DPInstrType::Marker) {}
  Instruction *MarkedInstr;
};

}

#endif // LLVM_IR_DEBUGPROGRAMINSTRUCTION_H
