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

static bool movePredicatesBeforeTerminators(MachineFunction &MF) {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    MachineBasicBlock::iterator FirstTerminator = MBB.getFirstTerminator();
    if (FirstTerminator == MBB.end())
      continue;

    SmallVector<MachineInstr *, 4> Predicates;
    for (MachineBasicBlock::iterator I = MBB.begin(); I != FirstTerminator; ++I)
      if (I->getOpcode() == RISCV::GPU_PRED)
        Predicates.push_back(&*I);

    for (MachineInstr *Predicate : Predicates) {
      MachineBasicBlock::iterator PredicateI = Predicate->getIterator();
      if (std::next(PredicateI) == FirstTerminator)
        continue;

      // SelectionDAG emits constants for successor PHIs while it lowers the
      // predecessor terminator. Keep these edge values under the old mask.
      MBB.splice(FirstTerminator, &MBB, PredicateI);
      Changed = true;
    }
  }
  return Changed;
}

bool GroomTMaskDependency::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getSubtarget<RISCVSubtarget>().hasExtGroom())
    return false;

  // GPU_PRED is inserted immediately before an IR branch. Instruction
  // selection can place successor-PHI materializations after it. Move the
  // predicate back next to the machine branch before tmask dependencies lock
  // the instruction order.
  bool Changed = movePredicatesBeforeTerminators(MF);
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
