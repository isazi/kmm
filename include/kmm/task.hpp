#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "kmm/block_header.hpp"
#include "kmm/event_list.hpp"
#include "kmm/identifiers.hpp"
#include "kmm/memory.hpp"

namespace kmm {

// Forward decl
class Runtime;
class RuntimeHandle;
class Device;
class Task;

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
 * Encapsulates the requirements for a task, including the device identifier and lists of inputs and outputs.
 */
struct TaskRequirements {
    TaskRequirements(DeviceId id) : device_id(id) {}

    DeviceId device_id;
    EventList dependencies;
    std::vector<TaskInput> inputs = {};
    std::vector<TaskOutput> outputs = {};
};

class TaskBuilder {
    struct Callback {
        virtual ~Callback() = default;
        virtual void call(EventId event_id) = 0;
    };

    template<typename F>
    struct CallbackImpl: Callback {
        CallbackImpl(F fun) : fun(std::move(fun)) {}

        void call(EventId event_id) {
            std::move(fun)(event_id);
        }

        F fun;
    };

  public:
    TaskBuilder(Runtime* runtime, DeviceId id);
    EventId submit(std::shared_ptr<Task> task);

    std::shared_ptr<Runtime> runtime() const;

    size_t add_input(BlockId block_id);
    size_t add_input(BlockId block_id, MemoryId memory_id);

    size_t add_output(std::unique_ptr<BlockHeader> header);
    size_t add_output(std::unique_ptr<BlockHeader> header, MemoryId memory_id);

    template<typename F>
    void after_submission(F fun) {
        m_callbacks.push_back(std::make_unique<CallbackImpl<F>>(std::move(fun)));
    }

  private:
    Runtime* m_runtime;
    TaskRequirements m_requirements;
    std::vector<std::unique_ptr<Callback>> m_callbacks;
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
    MemoryAllocation* allocation = nullptr;
};

/**
 * Provides the context required to run task, such as the input and output data.
 */
struct TaskContext {
    TaskContext() = default;

    std::vector<BlockAccessor> inputs;
    std::vector<BlockAccessorMut> outputs;
};

/**
 * Abstract class representing a task to be executed.
 */
class Task {
  public:
    virtual ~Task() = default;
    virtual void execute(Device&, TaskContext&) = 0;
};
}  // namespace kmm