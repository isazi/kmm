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
class WorkerState;

class WorkerJob: public Waker, public std::enable_shared_from_this<WorkerJob> {
  public:
    explicit WorkerJob(EventId id) : identifier(id) {}
    ~WorkerJob() override = default;

    EventId id() const {
        return identifier;
    }

    virtual void start(WorkerState&) {}
    virtual PollResult poll(WorkerState&) = 0;
    virtual void stop(WorkerState&) {}

    void trigger_wakeup() const final;
    void trigger_wakeup_and_poll() const;

  private:
    friend class Worker;
    friend class JobQueue;

    enum class Status { Created, Pending, Ready, Running, Done };
    Status status = Status::Created;
    EventId identifier;
    std::weak_ptr<Worker> worker;
    size_t unsatisfied_predecessors = 0;
    std::vector<std::shared_ptr<WorkerJob>> successors = {};
    std::atomic_flag in_queue = false;
    std::shared_ptr<WorkerJob> next_item = nullptr;
};

class JobQueue {
  public:
    JobQueue() = default;
    JobQueue(const JobQueue&) = delete;
    JobQueue(JobQueue&&) noexcept = default;

    JobQueue& operator=(const JobQueue&) = delete;
    JobQueue& operator=(JobQueue&&) noexcept = default;

    bool is_empty() const;
    bool push(std::shared_ptr<WorkerJob>);
    std::optional<std::shared_ptr<WorkerJob>> pop();

  private:
    std::shared_ptr<WorkerJob> m_head = nullptr;
    WorkerJob* m_tail = nullptr;
};

class SharedJobQueue {
  public:
    bool push(std::shared_ptr<WorkerJob>) const;
    JobQueue pop_all(std::chrono::time_point<std::chrono::system_clock> deadline = {}) const;

  private:
    mutable std::mutex m_lock;
    mutable std::condition_variable m_cond;
    mutable JobQueue m_queue;
};

class WorkerState {
  public:
    std::vector<std::shared_ptr<Executor>> executors;
    std::shared_ptr<MemoryManager> memory_manager;
    BlockManager block_manager;
};

class Worker: public std::enable_shared_from_this<Worker> {
  public:
    Worker(
        std::vector<std::shared_ptr<Executor>> executors,
        std::shared_ptr<MemoryManager> memory_manager);

    void make_progress(std::chrono::time_point<std::chrono::system_clock> deadline = {});
    void submit_command(EventId id, Command command, EventList dependencies);
    bool query_event(EventId id, std::chrono::time_point<std::chrono::system_clock> deadline = {});
    void wakeup(std::shared_ptr<WorkerJob> op, bool allow_progress = false);
    void shutdown();
    bool is_shutdown();

  private:
    void make_progress_impl(
        std::unique_lock<std::mutex> guard,
        std::chrono::time_point<std::chrono::system_clock> deadline = {});

    void satisfy_job_dependencies(std::shared_ptr<WorkerJob> op, size_t satisfied = 1);
    void start_job(std::shared_ptr<WorkerJob>& op);
    void poll_job(const std::shared_ptr<WorkerJob>& op);
    void stop_job(const std::shared_ptr<WorkerJob>& op);

    SharedJobQueue m_shared_poll_queue;

    std::mutex m_lock;
    std::condition_variable m_job_completion;
    std::unordered_map<EventId, std::shared_ptr<WorkerJob>> m_jobs;
    JobQueue m_poll_queue;
    JobQueue m_ready_queue;
    WorkerState m_state;
    bool m_shutdown = false;
};

}  // namespace kmm