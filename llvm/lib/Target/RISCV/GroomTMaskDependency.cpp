//===-- GroomTMaskDependency.cpp - Model GROOM tmask dependencies ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// GROOM lane-executed instructions implicitly read the tmask CSR. Model that
// dependency with the dummy TMASK physical register so that machine passes
// cannot move instructions across operations which update tmask.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "groom-tmask-dependency"
#define GROOM_TMASK_DEPENDENCY_NAME "GROOM tmask dependency"

namespace {

class GroomTMaskDependency : public MachineFunctionPass {
public:
  static char ID;

  GroomTMaskDependency() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return GROOM_TMASK_DEPENDENCY_NAME; }
};

} // end anonymous namespace

char GroomTMaskDependency::ID = 0;

INITIALIZE_PASS(GroomTMaskDependency, DEBUG_TYPE, GROOM_TMASK_DEPENDENCY_NAME,
                false, false)

bool GroomTMaskDependency::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getSubtarget<RISCVSubtarget>().hasExtGroom())
    return false;

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.isMetaInstruction() || MI.isPHI())
        continue;

      for (MachineOperand &MO : MI.operands()) {
        if (MO.isReg() && MO.isDef() && MO.getReg() == RISCV::TMASK &&
            MO.isDead()) {
          MO.setIsDead(false);
          Changed = true;
        }
      }

      if (!MI.readsRegister(RISCV::TMASK, /*TRI=*/nullptr)) {
        MI.addOperand(MachineOperand::CreateReg(RISCV::TMASK, /*IsDef=*/false,
                                                /*IsImp=*/true));
        Changed = true;
      }
    }
  }
  return Changed;
}

FunctionPass *llvm::createGroomTMaskDependencyPass() {
  return new GroomTMaskDependency();
}
