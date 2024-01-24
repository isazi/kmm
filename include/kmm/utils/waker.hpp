#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>

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

class ThreadWaker: public Waker {
  public:
    void trigger_wakeup(bool allow_progress) const override {
        std::lock_guard guard {m_mutex};
        m_notified = true;
        m_cond.notify_one();
    }

    void wait() const {
        std::unique_lock guard {m_mutex};
        while (!m_notified) {
            m_cond.wait(guard);
        }

        m_notified = false;
    }

  private:
    mutable std::mutex m_mutex;
    mutable std::condition_variable m_cond;
    mutable bool m_notified = false;
};

}  // namespace kmm
