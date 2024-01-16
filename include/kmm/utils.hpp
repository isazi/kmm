#pragma once

#include <memory>

#define KMM_NOT_COPYABLE(TYPE)             \
  public:                                  \
    TYPE(const TYPE&) = delete;            \
    TYPE& operator=(const TYPE&) = delete; \
    TYPE(TYPE&) = delete;                  \
    TYPE& operator=(TYPE&) = delete;       \
                                           \
  private:

#define KMM_NOT_COPYABLE_OR_MOVABLE(TYPE)            \
    KMM_NOT_COPYABLE(TYPE)                           \
  public:                                            \
    TYPE(TYPE&&) noexcept = delete;                  \
    TYPE& operator=(TYPE&&) noexcept = delete;       \
    TYPE(const TYPE&&) noexcept = delete;            \
    TYPE& operator=(const TYPE&&) noexcept = delete; \
                                                     \
  private:

namespace kmm {

using index_t = int;

enum class PollResult { Pending, Ready };

class Waker: public std::enable_shared_from_this<Waker> {
  public:
    virtual ~Waker() = default;
    virtual void trigger_wakeup(bool allow_progress) const = 0;

    void trigger_wakeup() const {
        trigger_wakeup(false);
    }
};

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
