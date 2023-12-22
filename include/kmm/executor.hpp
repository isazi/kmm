#pragma once

#include <memory>
#include <variant>
#include <vector>
#include <thread>

#include "kmm/memory.hpp"
#include "kmm/object.hpp"
#include "kmm/types.hpp"

namespace kmm {
struct BufferAccess {
    const MemoryAllocation* allocation = nullptr;
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

  private:
    std::shared_ptr<const std::string> m_reason;
};

using TaskResult = std::variant<std::monostate, ObjectHandle, TaskError>;

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

struct Job {
    virtual ~Job() = default;
    virtual void execute(ExecutorContext&) = 0;
    std::unique_ptr<Job> next;
};

struct Queue {
    std::mutex lock;
    std::condition_variable cond;
    bool has_shutdown = false;
    std::unique_ptr<Job> front = nullptr;
    Job* back = nullptr;

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

    void process_forever() {
        std::unique_lock<std::mutex> guard(lock);
        ExecutorContext context {};

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
};

}  // namespace kmm