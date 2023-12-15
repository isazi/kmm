#pragma once

#include <memory>
#include <variant>
#include <vector>

#include "kmm/block.hpp"
#include "kmm/memory.hpp"
#include "kmm/types.hpp"

namespace kmm {
struct InputBlock {
    BlockId block_id;
    std::shared_ptr<const BlockHeader> header;
    const MemoryAllocation* allocation = nullptr;
};

struct OutputBuffer {
    BlockId block_id;
    BlockHeader* header;
    const MemoryAllocation* allocation = nullptr;
};

struct TaskContext {
    std::vector<InputBlock> inputs;
    std::vector<OutputBuffer> outputs;
};

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

class TaskCompletion {
  public:
    class Impl {
      public:
        virtual ~Impl() = default;
        virtual void complete_task(TaskResult) = 0;
    };

    explicit TaskCompletion(std::shared_ptr<Impl> = {});
    TaskCompletion(TaskCompletion&&) noexcept = default;
    TaskCompletion(const TaskContext&) = delete;
    ~TaskCompletion();

    void complete(TaskResult);
    void complete_err(const std::string& error);

  private:
    std::shared_ptr<Impl> m_impl;
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
    virtual void submit(std::shared_ptr<Task>, TaskContext, TaskCompletion) = 0;
};

}  // namespace kmm