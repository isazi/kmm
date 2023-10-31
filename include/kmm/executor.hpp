#pragma once

#include <memory>
#include <vector>

#include "kmm/memory.hpp"
#include "kmm/types.hpp"

namespace kmm {
class Task;
class TaskCompletion;

struct BufferAccess {
    const Allocation* allocation = nullptr;
    bool writable = false;
};

class TaskContext {
    std::vector<BufferAccess> buffers;
    std::vector<std::shared_ptr<Object>> objects;
};

class Executor {
  public:
    virtual ~Executor() = default;
    virtual void submit(std::shared_ptr<Task>, TaskContext, TaskCompletion) = 0;
};

class ExecutorContext {};

class Task {
  public:
    virtual ~Task() = default;
    virtual void execute(ExecutorContext&, TaskContext&) = 0;
};

}  // namespace kmm