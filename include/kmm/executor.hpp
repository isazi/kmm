#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "kmm/block.hpp"
#include "kmm/memory.hpp"
#include "kmm/result.hpp"
#include "kmm/types.hpp"

namespace kmm {

/**
 * Represents an input for a task, including memory and block identifiers.
 */
struct TaskInput {
    MemoryId memory_id;
    BlockId block_id;
};

/**
 * Represents an output of a task, containing memory identifier and a block header.
 */
struct TaskOutput {
    MemoryId memory_id;
    std::unique_ptr<BlockHeader> header;
};

/**
 * Encapsulates the requirements for a task, including the executor identifier and lists of inputs and outputs.
 */
struct TaskRequirements {
    TaskRequirements(ExecutorId id) : executor_id(id) {}

    ExecutorId executor_id;
    std::vector<TaskInput> inputs = {};
    std::vector<TaskOutput> outputs = {};
};

class ITaskCompletion {
  public:
    virtual ~ITaskCompletion() = default;
    virtual void complete_task(Result<void>) = 0;
};

class TaskCompletion {
  public:
    explicit TaskCompletion(std::shared_ptr<ITaskCompletion> = {});
    ~TaskCompletion();

    TaskCompletion(TaskCompletion&&) noexcept = default;
    TaskCompletion& operator=(TaskCompletion&&) noexcept = default;

    TaskCompletion(const TaskCompletion&) = delete;
    TaskCompletion& operator=(const TaskCompletion&) = delete;

    void complete(Result<void> = {});

  private:
    std::shared_ptr<ITaskCompletion> m_impl;
};

/**
 * Provides read-only access to a block.
 */
struct BlockAccessor {
    BlockId block_id;
    std::shared_ptr<const BlockHeader> header;
    const MemoryAllocation* allocation = nullptr;
};

/**
 * Provides read-write access to a block.
 */
struct BlockAccessorMut {
    BlockId block_id;
    BlockHeader* header;
    const MemoryAllocation* allocation = nullptr;
};

struct TaskContext {
    TaskContext() = default;

    std::vector<BlockAccessor> inputs;
    std::vector<BlockAccessorMut> outputs;
};

/**
 * Represents the context in which an executor operates.
 */
class ExecutorContext {};

/**
 * Abstract class representing a task to be executed.
 */
class Task {
  public:
    virtual ~Task() = default;
    virtual void execute(ExecutorContext&, TaskContext&) = 0;
};

class Executor {
  public:
    virtual ~Executor() = default;
    virtual void submit(std::shared_ptr<Task>, TaskContext, TaskCompletion) = 0;
};

}  // namespace kmm