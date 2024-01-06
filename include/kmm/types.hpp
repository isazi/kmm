#pragma once

#include <memory>

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

}  // namespace kmm
