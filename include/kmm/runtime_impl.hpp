#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "kmm/event.hpp"
#include "kmm/executor.hpp"
#include "kmm/identifiers.hpp"
#include "kmm/worker/runner.hpp"

namespace kmm {

class Worker;

class RuntimeImpl: public std::enable_shared_from_this<RuntimeImpl> {
  public:
    RuntimeImpl(std::vector<std::shared_ptr<Executor>> executors, std::unique_ptr<Memory> memory);
    ~RuntimeImpl();

    EventId submit_task(std::shared_ptr<Task> task, TaskRequirements reqs, EventList deps = {})
        const;
    EventId delete_block(BlockId block_id, EventList deps = {}) const;
    EventId join_events(EventList deps) const;
    EventId submit_barrier() const;
    EventId submit_block_barrier(BlockId block_id) const;
    EventId submit_block_prefetch(BlockId block_id, MemoryId memory_id, EventList deps = {}) const;

    bool query_event(EventId id, std::chrono::time_point<std::chrono::system_clock> deadline = {})
        const;

  private:
    std::shared_ptr<Worker> m_worker;
    WorkerRunner m_thread;

    mutable std::mutex m_mutex;
    mutable uint64_t m_next_event = 1;
    mutable std::unordered_map<BlockId, EventList> m_block_accesses;
};

}  // namespace kmm