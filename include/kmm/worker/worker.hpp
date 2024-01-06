#pragma once

#include <chrono>
#include <memory>

#include "kmm/event_list.hpp"
#include "kmm/executor.hpp"
#include "kmm/identifiers.hpp"
#include "kmm/memory.hpp"
#include "kmm/worker/block_manager.hpp"
#include "kmm/worker/command.hpp"
#include "kmm/worker/job.hpp"
#include "kmm/worker/memory_manager.hpp"
#include "kmm/worker/scheduler.hpp"

namespace kmm {

class WorkerState {
  public:
    std::vector<std::shared_ptr<Executor>> executors;
    std::shared_ptr<MemoryManager> memory_manager;
    BlockManager block_manager;
};

class Worker: public std::enable_shared_from_this<Worker> {
  public:
    Worker(std::vector<std::shared_ptr<Executor>> executors, std::unique_ptr<Memory> memory);

    void make_progress(std::chrono::time_point<std::chrono::system_clock> deadline = {});
    void submit_command(EventId id, Command command, EventList dependencies);
    bool query_event(EventId id, std::chrono::time_point<std::chrono::system_clock> deadline = {});
    void wakeup(std::shared_ptr<CopyJob> job, bool allow_progress = false);
    void shutdown();
    bool is_shutdown();

  private:
    bool make_progress_impl();

    void start_job(std::shared_ptr<Scheduler::Node> node);
    void poll_job(CopyJob& job);
    void stop_job(CopyJob& job);

    std::shared_ptr<SharedJobQueue> m_shared_poll_queue;

    std::mutex m_lock;
    std::condition_variable m_job_completion;
    JobQueue m_local_poll_queue;
    Scheduler m_scheduler;
    WorkerState m_state;
    bool m_shutdown = false;
};

}  // namespace kmm