#include "kmm/utils/macros.hpp"

namespace kmm {

template<typename F>
class ScopeGuard {
    KMM_NOT_COPYABLE_OR_MOVABLE(ScopeGuard)

  public:
    ScopeGuard(F fun) : m_fun(std::move(fun)) {}
    ~ScopeGuard() {
        m_fun();
    }

  private:
    F m_fun;
};

template<typename F>
ScopeGuard(F) -> ScopeGuard<F>;

}  // namespace kmm