#pragma once

#include <condition_variable>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
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

template<typename Context>
class WorkQueue {
  public:
    class Job {
      public:
        virtual ~Job() = default;
        virtual void execute(Context&) = 0;

      private:
        friend WorkQueue<Context>;
        std::unique_ptr<Job> next;
    };

    void shutdown() {
        std::lock_guard<std::mutex> guard(lock);
        has_shutdown = true;
        cond.notify_all();
    }

    void push(std::unique_ptr<Job> unit) {
        std::lock_guard<std::mutex> guard(lock);
        if (front) {
            unit->next = nullptr;
            back->next = std::move(unit);
            back = back->next.get();
        } else {
            front = std::move(unit);
            back = front.get();
            cond.notify_all();
        }
    }

    void process_forever(Context& context) {
        std::unique_lock<std::mutex> guard(lock);

        while (!has_shutdown || front != nullptr) {
            if (front == nullptr) {
                cond.wait(guard);
                continue;
            }

            auto popped = std::move(front);
            if (popped->next != nullptr) {
                front = std::move(popped->next);
            } else {
                front = nullptr;
                back = nullptr;
            }

            guard.unlock();
            popped->execute(context);
            guard.lock();
        }
    }

  private:
    std::mutex lock;
    std::condition_variable cond;
    bool has_shutdown = false;
    std::unique_ptr<Job> front = nullptr;
    Job* back = nullptr;
};

}  // namespace kmm