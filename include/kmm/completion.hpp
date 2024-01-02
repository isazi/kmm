#pragma once

#include <memory>

#include "kmm/panic.hpp"
#include "kmm/result.hpp"

namespace kmm {

class CompletionHandler {
  public:
    virtual ~CompletionHandler() = default;
    virtual void complete(Result<void>) = 0;
};

class Completion {
  public:
    explicit Completion(std::shared_ptr<CompletionHandler> impl = {}) : m_impl(std::move(impl)) {}
    ~Completion() {
        if (m_impl) {
            KMM_PANIC("deleted `Completion` without calling `complete()`");
        }
    }

    Completion(Completion&&) noexcept = default;
    Completion& operator=(Completion&&) noexcept = default;

    Completion(const Completion&) = delete;
    Completion& operator=(const Completion&) = delete;

    explicit operator bool() const {
        return bool(m_impl);
    }

    void complete(Result<void> result) {
        KMM_ASSERT(m_impl != nullptr);
        auto impl = std::move(m_impl);
        impl->complete(std::move(result));
    }

    void complete_ok() {
        complete({});
    }

    template<typename E>
    void complete_error(E&& error) {
        complete(ErrorPtr(std::forward<E>(error)));
    }

  private:
    std::shared_ptr<CompletionHandler> m_impl;
};

}  // namespace kmm