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

    void wakeup_op(const std::shared_ptr<Operation>& op);
    void complete_op(const std::shared_ptr<Operation>& op);

  private:
    void trigger_predecessor_completed(const std::shared_ptr<Operation>& op, size_t count = 1);
    void stage_op(const std::shared_ptr<Operation>& op);
    void poll_op(const std::shared_ptr<Operation>& op);
    void schedule_op(const std::shared_ptr<Operation>& op);

    std::unordered_map<OperationId, std::shared_ptr<Operation>> m_ops;

    std::vector<std::shared_ptr<Executor>> m_executors;
    std::shared_ptr<MemoryManager> m_memory_manager;
    std::shared_ptr<ObjectManager> m_object_manager;

    std::mutex m_ready_lock;
    std::condition_variable m_ready_cond;
    std::deque<std::shared_ptr<Operation>> m_ready_ops;
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