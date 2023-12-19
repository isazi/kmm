#pragma once

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "kmm/executor.hpp"
#include "kmm/memory.hpp"
#include "kmm/types.hpp"
#include "kmm/worker/block_manager.hpp"
#include "kmm/worker/command.hpp"
#include "kmm/worker/memory_manager.hpp"
#include "kmm/worker/scheduler.hpp"

namespace kmm {

class Worker;
class WorkerState;

class Job: public Waker, public std::enable_shared_from_this<Job> {
  public:
    Job(EventId id) : identifier(id) {}
    Job(const Job&) = delete;
    Job(Job&&) = delete;

    virtual ~Job() = default;
    virtual void start(WorkerState&) {}
    virtual PollResult poll(WorkerState&) = 0;
    virtual void stop(WorkerState&) {}

    void trigger_wakeup(bool allow_progress) const final;

    EventId id() const {
        return identifier;
    }

  private:
    friend class Worker;
    friend class JobQueue;

    enum class Status { Created, Running, Done };
    Status status = Status::Created;

    std::atomic_flag in_queue = false;
    std::shared_ptr<Job> next_queue_item = nullptr;

    EventId identifier;
    std::weak_ptr<Worker> worker;
    std::shared_ptr<Scheduler::Node> completion = nullptr;
};

class JobQueue {
  public:
    JobQueue() = default;
    JobQueue(const JobQueue&) = delete;
    JobQueue(JobQueue&&) noexcept = default;

    JobQueue& operator=(const JobQueue&) = delete;
    JobQueue& operator=(JobQueue&&) noexcept = default;

    bool is_empty() const;
    bool push(std::shared_ptr<Job>);
    std::optional<std::shared_ptr<Job>> pop();

  private:
    std::shared_ptr<Job> m_head = nullptr;
    Job* m_tail = nullptr;
};

class SharedJobQueue {
  public:
    bool push(std::shared_ptr<Job>) const;
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
    void wakeup(std::shared_ptr<Job> job, bool allow_progress = false);
    void shutdown();
    bool is_shutdown();

  private:
    void make_progress_impl(
        std::unique_lock<std::mutex> guard,
        std::chrono::time_point<std::chrono::system_clock> deadline = {});

    void start_job(std::shared_ptr<Scheduler::Node> node);
    void poll_job(Job& job);
    void stop_job(Job& job);

    SharedJobQueue m_shared_poll_queue;

    std::mutex m_lock;
    std::condition_variable m_job_completion;
    JobQueue m_local_poll_queue;
    Scheduler m_scheduler;
    WorkerState m_state;
    bool m_shutdown = false;
};

}  // namespace kmm