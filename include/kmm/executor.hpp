#pragma once

#include <memory>
#include <variant>
#include <vector>

#include "kmm/block.hpp"
#include "kmm/memory.hpp"
#include "kmm/types.hpp"

namespace kmm {

class TaskError {
  public:
    TaskError(const char* error) : m_reason(std::make_shared<std::string>(error)) {}
    TaskError(const std::exception& e) : TaskError(e.what()) {}
    TaskError(const std::string& e) : TaskError(e.c_str()) {}

    const std::string& get() const {
        return *m_reason;
    }

  private:
    std::shared_ptr<const std::string> m_reason;
};

using TaskResult = std::variant<std::monostate, TaskError>;

struct BlockAccessor {
    BlockId block_id;
    std::shared_ptr<const BlockHeader> header;
    const MemoryAllocation* allocation = nullptr;
};

struct BlockAccessorMut {
    BlockId block_id;
    BlockHeader* header;
    const MemoryAllocation* allocation = nullptr;
};

class ITaskCompletion {
  public:
    virtual ~ITaskCompletion() = default;
    virtual void complete_task(TaskResult) = 0;
};

class TaskCompletion {
  public:
    explicit TaskCompletion(std::shared_ptr<ITaskCompletion> = {});
    TaskCompletion(TaskCompletion&&) noexcept = default;
    TaskCompletion(const TaskCompletion&) = delete;
    ~TaskCompletion();

    void complete(TaskResult);
    void complete_err(const std::string& error);

  private:
    std::shared_ptr<ITaskCompletion> m_impl;
};

struct TaskContext {
    TaskContext(TaskCompletion completion) : completion(std::move(completion)) {}

    std::vector<BlockAccessor> inputs;
    std::vector<BlockAccessorMut> outputs;
    TaskCompletion completion;
};

class ExecutorContext {};

class Task {
  public:
    virtual ~Task() = default;
    virtual TaskResult execute(ExecutorContext&, TaskContext&) = 0;
};

class Executor {
  public:
    virtual ~Executor() = default;
    virtual void submit(std::shared_ptr<Task>, TaskContext) = 0;
};

}  // namespace kmm