#pragma once

#include <memory>
#include <vector>

#include "fmt/format.h"

#include "kmm/block.hpp"
#include "kmm/event_list.hpp"
#include "kmm/identifiers.hpp"
#include "kmm/memory.hpp"

namespace kmm {

// Forward decl
class RuntimeImpl;

/**
 * Represents an input for a task specifying the memory and block identifier.
 */
struct TaskInput {
    BlockId block_id;
    std::optional<MemoryId> memory_id;
};

/**
 * Represents an output of a task, containing a memory identifier and a block header.
 */
struct TaskOutput {
    std::unique_ptr<BlockHeader> header;
    MemoryId memory_id;
};

/**
 * Encapsulates the requirements for a task, including the executor identifier and lists of inputs and outputs.
 */
struct TaskRequirements {
    TaskRequirements(ExecutorId id) : executor_id(id) {}

    size_t add_input(BlockId block_id);
    size_t add_input(BlockId block_id, MemoryId memory_id);
    size_t add_input(BlockId block_id, RuntimeImpl& rt);

    size_t add_output(std::unique_ptr<BlockHeader> header, MemoryId memory_id);
    size_t add_output(std::unique_ptr<BlockHeader> header, RuntimeImpl& rt);

    ExecutorId executor_id;
    EventList dependencies;
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
 * Represents information of an executor.
 */
class ExecutorInfo {
  public:
    virtual ~ExecutorInfo() = default;

    /**
     * The name of the executor. Useful for debugging.
     */
    virtual std::string name() const = 0;

    /**
     * Which memory does this executor has the strongest affinity to.
     */
    virtual MemoryId memory_affinity() const = 0;

    /**
     * Can the compute units from this executor access the specified memory?
     */
    virtual bool is_memory_accessible(MemoryId id) const {
        return id == memory_affinity();
    }
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

/**
 * Abstract class representing a compute unit that can process tasks.
 */
class Executor {
  public:
    virtual ~Executor() = default;
    virtual std::unique_ptr<ExecutorInfo> info() const = 0;
    virtual void submit(std::shared_ptr<Task>, TaskContext, Completion) const = 0;
};

}  // namespace kmm