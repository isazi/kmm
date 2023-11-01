#pragma once

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "kmm/command.hpp"
#include "kmm/executor.hpp"
#include "kmm/memory.hpp"
#include "kmm/memory_manager.hpp"
#include "kmm/object_manager.hpp"
#include "kmm/types.hpp"

namespace kmm {

class Scheduler: public std::enable_shared_from_this<Scheduler> {
  public:
    struct Operation;

    void make_progress(std::chrono::time_point<std::chrono::system_clock> deadline);
    void submit_command(CommandPacket packet);
    void wakeup(const std::shared_ptr<Operation>& op);
    void complete(const std::shared_ptr<Operation>& op);

  private:
    class ReadyQueue {
      public:
        void push(const std::shared_ptr<Operation>& op) const;
        void pop_all(
            std::chrono::time_point<std::chrono::system_clock> deadline,
            std::deque<std::shared_ptr<Operation>>& output) const;

      private:
        mutable std::mutex m_lock;
        mutable std::condition_variable m_cond;
        mutable std::deque<std::shared_ptr<Operation>> m_queue;
    };

    void make_progress_impl(
        std::chrono::time_point<std::chrono::system_clock> deadline,
        std::unique_lock<std::mutex> guard);
    void stage_op(const std::shared_ptr<Operation>& op);
    void poll_op(const std::shared_ptr<Operation>& op);
    void schedule_op(const std::shared_ptr<Operation>& op);
    void complete_op(const std::shared_ptr<Operation>& op);
    void trigger_predecessor_completed(const std::shared_ptr<Operation>& op, size_t count = 1);

    std::mutex m_lock;
    std::unordered_map<OperationId, std::shared_ptr<Operation>> m_ops;
    std::vector<std::shared_ptr<Executor>> m_executors;
    std::shared_ptr<MemoryManager> m_memory_manager;
    std::shared_ptr<ObjectManager> m_object_manager;

    ReadyQueue m_ready_queue;
};

class TaskCompletion {
  public:
    TaskCompletion(std::weak_ptr<Scheduler> worker, std::weak_ptr<Scheduler::Operation>);

    TaskCompletion(TaskCompletion&&) noexcept = default;
    TaskCompletion(const TaskContext&) = delete;
    void complete();
    ~TaskCompletion();

  private:
    std::weak_ptr<Scheduler> m_scheduler;
    std::weak_ptr<Scheduler::Operation> m_op;
};

}  // namespace kmm