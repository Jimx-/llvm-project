#include "GroomDivergenceTracker.h"
#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/IR/IntrinsicsRISCV.h"

#define CSR_WTID 0xcc0
#define CSR_LTID 0xcc1
#define CSR_GTID 0xcc2

#define CSR_RASTPOS 0xcd0
#define CSR_RASTPID 0xcd1
#define CSR_RASTBCA 0xcd2
#define CSR_RASTBCB 0xcd3
#define CSR_RASTBCC 0xcd4
#define CSR_RASTMASK 0xcd5

namespace groom {

using namespace llvm;

DivergenceTracker::DivergenceTracker(const Function &F)
    : m_function(&F), m_initialized(false) {}

void DivergenceTracker::initialize() { m_initialized = true; }

bool DivergenceTracker::eval(const Value *v) {
  if (!m_initialized) {
    initialize();
  }

  if (isa<AtomicRMWInst>(v) || isa<AtomicCmpXchgInst>(v)) {
    LLVM_DEBUG(dbgs() << "*** divergent atomic variable " << v->getName()
                      << "\n");
    return true;
  } else if (auto CI = dyn_cast<CallInst>(v)) {
    if (!CI->isInlineAsm())
      return false;

    auto CV = CI->getCalledOperand();
    if (const InlineAsm *IA = dyn_cast<InlineAsm>(CV)) {
      auto &asm_str = IA->getAsmString();

      if (asm_str.substr(0, 4) == "csrr") {
        auto AO = CI->getArgOperand(0);
        if (const ConstantInt *C = dyn_cast<ConstantInt>(AO)) {
          switch (C->getValue().extractBitsAsZExtValue(12, 0)) {
          case CSR_WTID:
          case CSR_LTID:
          case CSR_GTID:
          case CSR_RASTPOS:
          case CSR_RASTPID:
          case CSR_RASTBCA:
          case CSR_RASTBCB:
          case CSR_RASTBCC:
          case CSR_RASTMASK:

            LLVM_DEBUG(dbgs() << "*** divergent csr variable " << v->getName()
                              << "\n");
            m_dv.insert(v);
            return true;

          default:
            break;
          }
        }
      }
    }
  } else if (auto ST = dyn_cast<StoreInst>(v)) {
    auto addr = ST->getPointerOperand();
    if (dyn_cast<AllocaInst>(addr) != NULL) {
      auto value = ST->getValueOperand();
      if (m_dv.count(value)) {
        m_dv.insert(addr);
      }
    }
  } else if (auto LD = dyn_cast<LoadInst>(v)) {
    auto addr = LD->getPointerOperand();
    if (dyn_cast<AllocaInst>(addr) != NULL) {
      if (m_dv.count(addr)) {
        LLVM_DEBUG(dbgs() << "*** divergent load variable " << v->getName()
                          << "\n");
        m_dv.insert(v);
        return true;
      }
    }
  }

  return false;
}

} // namespace groom
