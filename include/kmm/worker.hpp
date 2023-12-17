#pragma once

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "kmm/block_manager.hpp"
#include "kmm/command.hpp"
#include "kmm/executor.hpp"
#include "kmm/memory.hpp"
#include "kmm/memory_manager.hpp"
#include "kmm/scheduler.hpp"
#include "kmm/types.hpp"

namespace kmm {

class Worker;

class WorkerJob: public Waker, public Job, public std::enable_shared_from_this<WorkerJob> {
  public:
    explicit WorkerJob(EventId id) : Job(id) {}

    void trigger_wakeup() const final;
    void trigger_wakeup_and_poll() const;

    void start(Worker&);
    virtual PollResult poll(Worker&) = 0;
    void stop(Worker&);

  public:
    bool is_running = false;
    bool is_ready = false;
    std::weak_ptr<Worker> m_worker;
};

class ExecuteJob: public WorkerJob, public TaskCompletion::Impl {
  public:
    ExecuteJob(EventId id, CommandExecute command) :
        WorkerJob(id),
        device_id(command.device_id),
        task(std::move(command.task)),
        inputs(std::move(command.inputs)),
        outputs(std::move(command.outputs)) {}

    PollResult poll(Worker&) final;
    void complete_task(TaskResult result) final;

  private:
    enum class Status { Created, Staging, Running, Done };
    Status status = Status::Created;

    DeviceId device_id;
    std::shared_ptr<Task> task;
    std::vector<TaskInput> inputs;
    std::vector<TaskOutput> outputs;
    std::vector<std::optional<BufferId>> output_buffers;
    std::vector<MemoryRequest> memory_requests = {};
    std::optional<TaskResult> result = std::nullopt;
};

class DeleteBlockJob: public WorkerJob {
  public:
    DeleteBlockJob(EventId id, BlockId block_id) : WorkerJob(id), block_id(block_id) {}
    PollResult poll(Worker&) final;

  private:
    BlockId block_id;
};

class EmptyJob: public WorkerJob {
  public:
    EmptyJob(EventId id) : WorkerJob(id) {}
    PollResult poll(Worker&) final;
};

class Worker: public std::enable_shared_from_this<Worker> {
  public:
    Worker(
        std::vector<std::shared_ptr<Executor>> executors,
        std::unique_ptr<MemoryManager> memory_manager);

    void make_progress(std::chrono::time_point<std::chrono::system_clock> deadline = {});
    void submit_command(EventId id, Command command, EventList dependencies);
    bool query_event(EventId id, std::chrono::time_point<std::chrono::system_clock> deadline = {});
    void wakeup(std::shared_ptr<WorkerJob> op, bool allow_progress = false);
    void shutdown();
    bool has_shutdown();

  private:
    friend class ExecuteJob;
    friend class DeleteBlockJob;
    friend class TaskCompletion;

    void make_progress_impl(
        std::unique_lock<std::mutex> guard,
        std::chrono::time_point<std::chrono::system_clock> deadline = {});
    void poll_job(const std::shared_ptr<WorkerJob>& op);

    class WorkQueue {
      public:
        void push(std::shared_ptr<WorkerJob> op) const;
        void pop_nonblocking(std::deque<std::shared_ptr<WorkerJob>>& output) const;
        void pop_blocking(
            std::chrono::time_point<std::chrono::system_clock> deadline,
            std::deque<std::shared_ptr<WorkerJob>>& output) const;

      private:
        void pop_impl(std::deque<std::shared_ptr<WorkerJob>>& output) const;

        mutable std::mutex m_lock;
        mutable std::condition_variable m_cond;
        mutable std::deque<std::shared_ptr<WorkerJob>> m_queue;
    };

    WorkQueue m_poll_queue;

    std::mutex m_lock;
    mutable std::deque<std::shared_ptr<WorkerJob>> m_queue;
    std::vector<std::shared_ptr<Executor>> m_executors;
    std::unique_ptr<MemoryManager> m_memory_manager;
    BlockManager m_block_manager;
    Scheduler m_scheduler;
    bool m_shutdown = false;
};

}  // namespace kmm