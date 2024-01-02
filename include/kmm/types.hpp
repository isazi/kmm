#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "fmt/format.h"

#include "kmm/event.hpp"
#include "kmm/identifiers.hpp"

namespace kmm {

using index_t = int;

enum class PollResult { Pending, Ready };

class Waker: public std::enable_shared_from_this<Waker> {
  public:
    virtual ~Waker() = default;
    virtual void trigger_wakeup(bool allow_progress = false) const = 0;
};

}  // namespace kmm
