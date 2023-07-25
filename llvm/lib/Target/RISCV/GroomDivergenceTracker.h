#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Value.h"

namespace groom {

using namespace llvm;

class DivergenceTracker {
private:
  const Function *m_function;
  DenseSet<const Value *> m_dv;
  bool m_initialized;

  void initialize();

public:
  DivergenceTracker(const Function &F);

  bool eval(const Value *v);
};

} // namespace groom
