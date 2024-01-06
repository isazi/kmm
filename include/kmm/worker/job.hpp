#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>

#include "kmm/identifiers.hpp"
#include "kmm/types.hpp"
#include "kmm/worker/scheduler.hpp"

namespace kmm {

class Worker;
class WorkerState;

class CopyJob: public Waker {
  public:
    CopyJob(EventId id) : identifier(id) {}
    CopyJob(const CopyJob&) = delete;
    CopyJob(CopyJob&&) = delete;
    ~CopyJob() override = default;

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
    std::shared_ptr<CopyJob> next_queue_item = nullptr;

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
    bool push(std::shared_ptr<CopyJob>);
    void push_all(JobQueue);
    std::optional<std::shared_ptr<CopyJob>> pop();

  private:
    std::shared_ptr<CopyJob> m_head = nullptr;
    CopyJob* m_tail = nullptr;
};

class SharedJobQueue {
  public:
    bool push_job(std::shared_ptr<CopyJob>) const;
    JobQueue pop_all_jobs() const;

    bool wait_until(std::chrono::time_point<std::chrono::system_clock> deadline) const;

    mutable std::mutex m_lock;
    mutable bool m_needs_processing = false;
    mutable std::condition_variable m_cond;
    mutable JobQueue m_queue;
};
}  // namespace kmm