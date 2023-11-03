#pragma once

#include <memory>
#include <variant>
#include <vector>

#include "kmm/memory.hpp"
#include "kmm/object.hpp"
#include "kmm/types.hpp"

namespace kmm {
class Task;
class TaskCompletion;

struct BufferAccess {
    const Allocation* allocation = nullptr;
    bool writable = false;
};

struct TaskContext {
    std::vector<BufferAccess> buffers;
    std::vector<ObjectHandle> objects;
};

class TaskError {
  public:
    TaskError(const std::string& error = {}) : m_reason(std::make_shared<std::string>(error)) {}

    const std::string& get() const {
        return *m_reason;
    }

    std::shared_ptr<const std::string> m_reason;
};

using TaskResult = std::variant<std::monostate, ObjectHandle, TaskError>;

class Executor {
  public:
    virtual ~Executor() = default;
    virtual void submit(std::shared_ptr<Task>, TaskContext, TaskCompletion) = 0;
};

class ExecutorContext {};

class Task {
  public:
    virtual ~Task() = default;
    virtual TaskResult execute(ExecutorContext&, TaskContext&) = 0;
};

}  // namespace kmm