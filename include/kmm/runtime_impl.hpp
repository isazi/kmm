#pragma once

#include <mutex>

#include "kmm/worker_thread.hpp"

namespace kmm {

class Worker;

class RuntimeImpl: public std::enable_shared_from_this<RuntimeImpl> {
  public:
    RuntimeImpl(std::vector<std::shared_ptr<Executor>> executors, std::shared_ptr<Memory> memory);
    ~RuntimeImpl();

    EventId submit_task(std::shared_ptr<Task> task, TaskRequirements reqs, EventList deps = {})
        const;
    EventId delete_block(BlockId id, EventList deps = {}) const;
    EventId join_events(EventList deps) const;
    EventId submit_barrier() const;

    bool query_event(EventId id, std::chrono::time_point<std::chrono::system_clock> deadline = {})
        const;

  private:
    std::shared_ptr<Worker> m_worker;
    WorkerThread m_thread;

    mutable std::mutex m_mutex;
    mutable uint64_t m_next_event = 1;
    mutable std::unordered_map<BlockId, EventList> m_block_accesses;
};

}  // namespace kmm