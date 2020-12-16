//===- LiveDebugValues.cpp - Tracking Debug Value MIs ---------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_LIVEDEBUGVALUES_LIVEDEBUGVALUES_H
#define LLVM_LIB_CODEGEN_LIVEDEBUGVALUES_LIVEDEBUGVALUES_H

#include "llvm/CodeGen/DbgEntityHistoryCalculator.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetPassConfig.h"

namespace llvm {
class LiveDebugValues;

// Inline namespace for types / symbols shared between different
// LiveDebugValues implementations.
inline namespace SharedLiveDebugValues {

class LDVHistoryMaps {
public:
  DbgValueHistoryMap DbgValueMap;
  DbgLabelInstrMap LabelMap;

  LDVHistoryMaps() { }
  LDVHistoryMaps(const LDVHistoryMaps &) = delete;
  LDVHistoryMaps(const LDVHistoryMaps &&Other)
    : DbgValueMap(std::move(Other.DbgValueMap)),
                  LabelMap(std::move(Other.LabelMap)) { }

  LDVHistoryMaps &operator=(const LDVHistoryMaps &) = delete;
  LDVHistoryMaps &operator=(const LDVHistoryMaps &&Other) {
    DbgValueMap = std::move(Other.DbgValueMap);
    LabelMap = std::move(Other.LabelMap);
    return *this;
  }
};

// Expose a base class for LiveDebugValues interfaces to inherit from. This
// allows the generic LiveDebugValues pass handles to call into the
// implementation.
class LDVImpl {
public:
  virtual bool ExtendRanges(MachineFunction &MF, TargetPassConfig *TPC) = 0;
  virtual LDVHistoryMaps ExtendRangesAndCalculateHistory(MachineFunction &MF, TargetPassConfig *TPC) = 0;
  virtual ~LDVImpl() {}
};

} // namespace SharedLiveDebugValues

// Factory functions for LiveDebugValues implementations.
extern LDVImpl *makeVarLocBasedLiveDebugValues();
extern LDVImpl *makeInstrRefBasedLiveDebugValues();

/// Generic LiveDebugValues pass. Calls through to VarLocBasedLDV or
/// InstrRefBasedLDV to perform location propagation, via the LDVImpl
/// base class.
class LiveDebugValues : public MachineFunctionPass {
public:
  static char ID;

  LiveDebugValues();
  ~LiveDebugValues() {
    if (TheImpl)
      delete TheImpl;
  }

  /// Calculate the liveness information for the given machine function.
  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  LDVHistoryMaps getMaps() {
    LDVHistoryMaps Cpy = std::move(Maps); // So going to break on MSVC.
    Maps = LDVHistoryMaps();
    return Cpy;
  }

private:
  LDVImpl *TheImpl;
  TargetPassConfig *TPC;
  LDVHistoryMaps Maps;
};

} // namespace llvm

#endif // LLVM_LIB_CODEGEN_LIVEDEBUGVALUES_LIVEDEBUGVALUES_H

