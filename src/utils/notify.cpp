#include "kmm/utils/notify.hpp"

namespace kmm {

NotifyHandle::NotifyHandle(std::shared_ptr<Notify> m) : m_impl(std::move(m)) {}

NotifyHandle::NotifyHandle(std::unique_ptr<Notify> m) : m_impl(std::move(m)) {}

NotifyHandle::~NotifyHandle() {
    notify_and_clear();
}

void NotifyHandle::notify() noexcept {
    if (m_impl) {
        m_impl->notify();
    }
}

void NotifyHandle::clear() noexcept {
    m_impl = nullptr;
}

void NotifyHandle::notify_and_clear() noexcept {
    if (auto m = std::exchange(m_impl, nullptr)) {
        m->notify();
    }
}

}  // namespace kmm