#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsRISCV.h"

#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LegacyDivergenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/RegionIterator.h"
#include "llvm/Analysis/RegionPass.h"
#include "llvm/Analysis/TargetTransformInfo.h"

using namespace llvm;

namespace groom {

class GroomBranchDivergence : public FunctionPass {

  std::vector<BasicBlock *> m_div_bbs;
  DenseSet<BasicBlock *> m_div_bb_set;

  std::vector<Loop *> m_loops;
  DenseSet<Loop *> m_loop_set;

  LegacyDivergenceAnalysis *m_DA;
  DominatorTree *m_DT;
  PostDominatorTree *m_PDT;
  LoopInfo *m_LI;
  RegionInfo *m_RI;

  Type *m_sizet_ty;

  Function *m_tmask_func;
  Function *m_tmc_func;
  Function *m_pred_func;
  Function *m_split_func;
  Function *m_join_func;

  void initialize(Function &F, const RISCVSubtarget &ST);

  void processBranches(LLVMContext *context, Function *function);
  void processLoops(LLVMContext *context, Function *function);

  bool isUniform(BranchInst *T);

public:
  static char ID;

  GroomBranchDivergence();

  StringRef getPassName() const override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnFunction(Function &F) override;
};

} // namespace groom

using groom::GroomBranchDivergence;

namespace llvm {

void initializeGroomBranchDivergencePass(PassRegistry &);

FunctionPass *createGroomBranchDivergencePass() {
  return new GroomBranchDivergence();
}

} // namespace llvm

INITIALIZE_PASS_BEGIN(GroomBranchDivergence, "groom-branch-divergence",
                      "Groom Branch Divergence", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(RegionInfoPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LegacyDivergenceAnalysis)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(GroomBranchDivergence, "groom-branch-divergence",
                    "Groom Branch Divergence", false, false)

namespace groom {

char GroomBranchDivergence::ID = 0;

GroomBranchDivergence::GroomBranchDivergence() : FunctionPass(ID) {
  initializeGroomBranchDivergencePass(*PassRegistry::getPassRegistry());
}

StringRef GroomBranchDivergence::getPassName() const {
  return "Groom Split/Join Insertion";
}

void GroomBranchDivergence::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<RegionInfoPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<PostDominatorTreeWrapperPass>();
  AU.addRequired<LegacyDivergenceAnalysis>();
  AU.addRequired<TargetPassConfig>();
  FunctionPass::getAnalysisUsage(AU);
}

void GroomBranchDivergence::initialize(Function &F, const RISCVSubtarget &ST) {
  auto &M = *F.getParent();
  auto &Context = M.getContext();

  auto ptr_size = M.getDataLayout().getPointerSizeInBits();
  switch (ptr_size) {
  case 128:
    m_sizet_ty = llvm::Type::getInt128Ty(Context);
    break;
  case 64:
    m_sizet_ty = llvm::Type::getInt64Ty(Context);
    break;
  case 32:
    m_sizet_ty = llvm::Type::getInt32Ty(Context);
    break;
  case 16:
    m_sizet_ty = llvm::Type::getInt16Ty(Context);
    break;
  case 8:
    m_sizet_ty = llvm::Type::getInt8Ty(Context);
    break;
  default:
    LLVM_DEBUG(dbgs() << "Error: invalid pointer size: " << ptr_size << "\n");
  }

  m_tmask_func =
      Intrinsic::getDeclaration(&M, Intrinsic::riscv_gpu_tmask, {m_sizet_ty});
  m_tmc_func =
      Intrinsic::getDeclaration(&M, Intrinsic::riscv_gpu_tmc, {m_sizet_ty});
  m_pred_func =
      Intrinsic::getDeclaration(&M, Intrinsic::riscv_gpu_pred, {m_sizet_ty});
  m_split_func =
      Intrinsic::getDeclaration(&M, Intrinsic::riscv_gpu_split, {m_sizet_ty});
  m_join_func = Intrinsic::getDeclaration(&M, Intrinsic::riscv_gpu_join);

  m_RI = &getAnalysis<RegionInfoPass>().getRegionInfo();
  m_LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  m_DA = &getAnalysis<LegacyDivergenceAnalysis>();
  m_DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  m_PDT = &getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();

  m_div_bbs.clear();
  m_div_bb_set.clear();
}

bool GroomBranchDivergence::runOnFunction(Function &F) {
  LLVM_DEBUG(dbgs() << "*** Groom Branch Divergence ***\n");

  const auto &TPC = getAnalysis<TargetPassConfig>();
  const auto &TM = TPC.getTM<TargetMachine>();
  const auto &ST = TM.getSubtarget<RISCVSubtarget>(F);

  initialize(F, ST);

  auto &Context = F.getContext();

  for (auto I = df_begin(&F.getEntryBlock()), E = df_end(&F.getEntryBlock());
       I != E; ++I) {
    auto BB = *I;

    auto Br = dyn_cast<BranchInst>(BB->getTerminator());
    if (!Br)
      continue;

    if (Br->isUnconditional()) {
      LLVM_DEBUG(dbgs() << "*** skip non-conditional branch: " << BB->getName()
                        << "\n");
      continue;
    }

    if (isUniform(Br)) {
      LLVM_DEBUG(dbgs() << "*** skip uniform branch: " << BB->getName()
                        << "\n");
      continue;
    }

    auto loop = m_LI->getLoopFor(BB);
    if (loop) {
      auto ipdom = m_PDT->findNearestCommonDominator(Br->getSuccessor(0),
                                                     Br->getSuccessor(1));
      if (ipdom && loop->contains(ipdom)) {
        if (m_div_bb_set.insert(BB).second) {
          LLVM_DEBUG(dbgs() << "*** adding divergent branch: " << BB->getName()
                            << "\n");
          m_div_bbs.push_back(BB);
        }
      } else {
        if (m_loop_set.insert(loop).second) {
          LLVM_DEBUG(dbgs()
                     << "*** adding divergent loop: " << BB->getName() << "\n");
          m_loops.push_back(loop);
        }
      }
    } else {
      auto ipdom = m_PDT->findNearestCommonDominator(Br->getSuccessor(0),
                                                     Br->getSuccessor(1));
      if (!ipdom) {
        llvm::errs() << "*** skip divergent branch with no IPDOM: "
                     << BB->getName() << "\n";
        continue;
      }

      if (m_div_bb_set.insert(BB).second) {
        LLVM_DEBUG(dbgs() << "*** adding divergent branch: " << BB->getName()
                          << "\n");
        m_div_bbs.push_back(BB);
      }
    }
  }

  if (!m_loops.empty() || !m_div_bbs.empty()) {
    if (!m_loops.empty()) {
      processLoops(&Context, &F);
      m_loops.clear();
      m_PDT->recalculate(F);
    }

    if (!m_div_bbs.empty()) {
      processBranches(&Context, &F);
      m_div_bbs.clear();
    }
  }

  return true;
}

static void findSuccessor(DenseSet<BasicBlock *> &visited, BasicBlock *current,
                          BasicBlock *target, std::vector<BasicBlock *> &out) {
  visited.insert(current);
  auto branch = dyn_cast<BranchInst>(current->getTerminator());
  if (!branch)
    return;
  for (auto succ : branch->successors()) {
    if (succ == target) {
      out.push_back(current);
    } else {
      if (visited.count(succ) == 0) {
        findSuccessor(visited, succ, target, out);
      }
    }
  }
}

static void findSuccessor(BasicBlock *start, BasicBlock *target,
                          std::vector<BasicBlock *> &out) {
  DenseSet<BasicBlock *> visited;
  findSuccessor(visited, start, target, out);
}

static void insertBasicBlock(const std::vector<BasicBlock *> &BBs,
                             BasicBlock *succBB, BasicBlock *newBB) {
  DenseMap<std::pair<PHINode *, BasicBlock *>, PHINode *> phi_table;
  for (auto BB : BBs) {
    auto TI = BB->getTerminator();
    TI->replaceSuccessorWith(succBB, newBB);
    for (auto &I : *succBB) {
      auto phi = dyn_cast<PHINode>(&I);
      if (!phi)
        continue;
      for (unsigned op = 0, n = phi->getNumOperands(); op != n; ++op) {
        if (phi->getIncomingBlock(op) != BB)
          continue;
        PHINode *phi_stub;
        auto key = std::make_pair(phi, newBB);
        auto entry = phi_table.find(key);
        if (entry != phi_table.end()) {
          phi_stub = entry->second;
        } else {
          phi_stub = PHINode::Create(phi->getType(), 1, phi->getName(),
                                     &newBB->front());
          phi_table[key] = phi_stub;
          phi->addIncoming(phi_stub, newBB);
        }
        auto value = phi->removeIncomingValue(op);
        phi_stub->addIncoming(value, BB);
      }
    }
  }
}

void GroomBranchDivergence::processBranches(LLVMContext *context,
                                            Function *function) {
  std::unordered_map<BasicBlock *, BasicBlock *> ipdoms;

  for (auto BI = m_div_bbs.rbegin(), BIE = m_div_bbs.rend(); BI != BIE; ++BI) {
    auto BB = *BI;
    auto Br = dyn_cast<BranchInst>(BB->getTerminator());
    auto ipdom = m_PDT->findNearestCommonDominator(Br->getSuccessor(0),
                                                   Br->getSuccessor(1));
    ipdoms.insert(std::make_pair(BB, ipdom));
  }

  for (auto BI = m_div_bbs.rbegin(), BIE = m_div_bbs.rend(); BI != BIE; ++BI) {
    auto BB = *BI;
    auto Br = dyn_cast<BranchInst>(BB->getTerminator());
    auto ipdom = ipdoms[BB];
    bool is_sfb =
        ipdom == Br->getSuccessor(0) ||
        (ipdom == Br->getSuccessor(1) && m_div_bb_set.count(ipdom) == 0);

    IRBuilder<> ir_builder(Br);
    BasicBlock *stub = nullptr;
    auto cond = Br->getCondition();

    if (is_sfb) {
      bool inv_cond = ipdom == Br->getSuccessor(0);

      LLVM_DEBUG(dbgs() << "*** save tmask before divergent branch: "
                        << BB->getName() << "\n");
      auto tmask = CallInst::Create(m_tmask_func, "tmask", Br);

      if (inv_cond) {
        cond = ir_builder.CreateNot(cond, cond->getName() + ".not");
      }
      auto cond_cast = ir_builder.CreateIntCast(cond, m_sizet_ty, false,
                                                cond->getName() + ".i32");
      LLVM_DEBUG(dbgs() << "*** insert predicate before divergent branch: "
                        << BB->getName() << "\n");
      CallInst::Create(m_pred_func, cond_cast, "", Br);

      stub = BasicBlock::Create(*context, "join_stub", function, ipdom);
      LLVM_DEBUG(dbgs() << "*** restore tmask before IPDOM: "
                        << ipdom->getName() << "\n");
      auto stub_br = BranchInst::Create(ipdom, stub);
      CallInst::Create(m_tmc_func, tmask, "", stub_br);
    } else {
      auto cond_cast = ir_builder.CreateIntCast(cond, m_sizet_ty, false,
                                                cond->getName() + ".i32");
      LLVM_DEBUG(dbgs() << "*** insert split before divergent branch: "
                        << BB->getName() << "\n");
      CallInst::Create(m_split_func, cond_cast, "", Br);

      stub = BasicBlock::Create(*context, "join_stub", function, ipdom);
      LLVM_DEBUG(dbgs() << "*** insert join stub before IPDOM: "
                        << ipdom->getName() << "\n");
      auto stub_br = BranchInst::Create(ipdom, stub);
      CallInst::Create(m_join_func, "", stub_br);
    }

    std::vector<BasicBlock *> succs;
    findSuccessor(BB, ipdom, succs);
    insertBasicBlock(succs, ipdom, stub);
  }
}

void GroomBranchDivergence::processLoops(LLVMContext *context,
                                         Function *function) {
  DenseSet<const BasicBlock *> stub_blocks;

  for (auto it = m_loops.rbegin(); it != m_loops.rend(); ++it) {
    auto loop = *it;

    auto preheader = loop->getLoopPreheader();
    auto preheader_br = dyn_cast<BranchInst>(preheader->getTerminator());

    LLVM_DEBUG(dbgs() << "*** save tmask before preheader branch: "
                      << preheader->getName() << "\n");
    auto tmask = CallInst::Create(m_tmask_func, "tmask", preheader_br);

    SmallVector<BasicBlock *, 8> exiting_blocks;
    loop->getExitingBlocks(exiting_blocks);

    for (auto &p : exiting_blocks) {
      auto br = dyn_cast<BranchInst>(p->getTerminator());

      for (auto succ : br->successors()) {
        if (loop->contains(succ) || stub_blocks.count(succ) != 0)
          continue;

        IRBuilder<> ir_builder(br);
        auto cond = br->getCondition();
        auto cond_not = ir_builder.CreateNot(cond, cond->getName() + ".not");
        auto cond_not_cast = ir_builder.CreateIntCast(
            cond_not, m_sizet_ty, false, cond_not->getName() + ".i32");
        LLVM_DEBUG(dbgs() << "*** insert predicate before loop exit: "
                          << p->getName() << "\n");
        CallInst::Create(m_pred_func, cond_not_cast, "", br);

        auto stub =
            BasicBlock::Create(*context, "loop_exit_stub", function, succ);
        LLVM_DEBUG(dbgs() << "*** restore tmask before loop exit: "
                          << succ->getName() << "\n");
        stub_blocks.insert(stub);
        auto stub_br = BranchInst::Create(succ, stub);
        CallInst::Create(m_tmc_func, tmask, "", stub_br);
        insertBasicBlock({p}, succ, stub);
      }
    }
  }
}

bool GroomBranchDivergence::isUniform(BranchInst *T) {
  return m_DA->isUniform(T) ||
         (T->getMetadata("structurizecfg.uniform") != nullptr);
}

} // namespace groom
