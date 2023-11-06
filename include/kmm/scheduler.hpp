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
    friend class TaskCompletion;
    struct Operation;

    Scheduler(
        std::vector<std::shared_ptr<Executor>> executors,
        std::shared_ptr<MemoryManager> memory_manager,
        std::shared_ptr<ObjectManager> object_manager);

    void make_progress(
        std::optional<std::chrono::time_point<std::chrono::system_clock>> deadline = {});
    void submit_command(CommandPacket packet);
    void wakeup(const std::shared_ptr<Operation>& op);
    void shutdown();
    bool has_shutdown();

  protected:
    void complete(const std::shared_ptr<Operation>& op, TaskResult result);

  private:
    void make_progress_impl(
        std::unique_lock<std::mutex> guard,
        std::optional<std::chrono::time_point<std::chrono::system_clock>> deadline = {});
    void stage_op(const std::shared_ptr<Operation>& op);
    void poll_op(const std::shared_ptr<Operation>& op);
    void schedule_op(const std::shared_ptr<Operation>& op);
    void complete_op(const std::shared_ptr<Operation>& op, TaskResult result);
    void trigger_predecessor_completed(const std::shared_ptr<Operation>& op, size_t count = 1);

    class ReadyQueue {
      public:
        void push(std::shared_ptr<Operation> op) const;
        void pop_nonblocking(std::deque<std::shared_ptr<Operation>>& output) const;
        void pop_blocking(
            std::chrono::time_point<std::chrono::system_clock> deadline,
            std::deque<std::shared_ptr<Operation>>& output) const;

      private:
        void pop_impl(std::deque<std::shared_ptr<Operation>>& output) const;

        mutable std::mutex m_lock;
        mutable std::condition_variable m_cond;
        mutable std::deque<std::shared_ptr<Operation>> m_queue;
    };

    ReadyQueue m_ready_queue;

    std::mutex m_lock;
    std::unordered_map<OperationId, std::shared_ptr<Operation>> m_ops;
    std::vector<std::shared_ptr<Executor>> m_executors;
    std::shared_ptr<MemoryManager> m_memory_manager;
    std::shared_ptr<ObjectManager> m_object_manager;
    bool m_shutdown = false;
};

}  // namespace kmm