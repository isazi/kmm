#pragma once

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "spdlog/spdlog.h"

#include "kmm/command.hpp"
#include "kmm/executor.hpp"
#include "kmm/memory_manager.hpp"
#include "kmm/utils.hpp"

namespace kmm {

class Job {
  public:
    enum class Status { Created, Pending, Running, Done };

    explicit Job(EventId id) : identifier(id) {}
    virtual ~Job() = default;

    EventId id() const {
        return identifier;
    }

  private:
    friend class Scheduler;

    EventId identifier;
    Status status = Status::Created;
    size_t unsatisfied_predecessors = 0;
    std::vector<std::shared_ptr<Job>> successors = {};
};

class Scheduler {
  public:
    void insert_job(std::shared_ptr<Job> node, EventList dependencies);
    std::optional<std::shared_ptr<Job>> pop_ready_job();
    void mark_job_complete(EventId id);
    bool all_complete() const;
    bool is_job_complete(EventId id) const;
    bool is_job_complete_with_deadline(
        EventId id,
        std::unique_lock<std::mutex>& guard,
        std::chrono::time_point<std::chrono::system_clock> deadline);

  private:
    void trigger_predecessor_completed(const std::shared_ptr<Job>& op, size_t count = 1);

    std::unordered_map<EventId, std::shared_ptr<Job>> m_jobs;
    std::condition_variable m_completion_condvar;
    std::deque<std::shared_ptr<Job>> m_ready_queue;
};

}  // namespace kmm