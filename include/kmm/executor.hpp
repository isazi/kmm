#pragma once

#include <condition_variable>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "kmm/block.hpp"
#include "kmm/completion.hpp"
#include "kmm/memory.hpp"
#include "kmm/result.hpp"
#include "kmm/types.hpp"

namespace kmm {

/**
 * Represents an input for a task specifying the memory and block identifier.
 */
struct TaskInput {
    MemoryId memory_id;
    BlockId block_id;
};

/**
 * Represents an output of a task, containing a memory identifier and a block header.
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
    virtual void submit(std::shared_ptr<Task>, TaskContext, Completion) = 0;
};

}  // namespace kmm