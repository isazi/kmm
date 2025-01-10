#pragma once

#include <vector>

#include "kmm/core/buffer.hpp"
#include "kmm/core/execution_context.hpp"

namespace kmm {

enum struct ExecutionSpace { Host, Device };

struct TaskContext {
    std::vector<BufferAccessor> accessors;
};

class Task {
  public:
    virtual ~Task() = default;
    virtual void execute(ExecutionContext& device, TaskContext context) = 0;
};

}  // namespace kmm