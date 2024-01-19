#pragma once

#include <chrono>
#include <memory>

#include "kmm/device.hpp"
#include "kmm/event_list.hpp"
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
    std::vector<std::shared_ptr<DeviceHandle>> devices;
    std::shared_ptr<MemoryManager> memory_manager;
    BlockManager block_manager;
};

struct BlockReadGuard {
    std::shared_ptr<Worker> worker;
    MemoryRequest request;
    std::shared_ptr<BlockHeader> header;
    const MemoryAllocation* alloc;
};

struct BlockWriteGuard {
    std::shared_ptr<Worker> worker;
    MemoryRequest request;
    std::shared_ptr<BlockHeader> header;
    MemoryAllocation* alloc;
};

class Worker: public std::enable_shared_from_this<Worker> {
  public:
    Worker(std::vector<std::shared_ptr<DeviceHandle>> devices, std::unique_ptr<Memory> memory);

    void make_progress(std::chrono::time_point<std::chrono::system_clock> deadline = {});

    void create_block(
        BlockId block_id,
        MemoryId memory_id,
        std::unique_ptr<BlockHeader> header,
        const void* src_data,
        size_t num_bytes);
    std::shared_ptr<BlockHeader> read_block(
        BlockId block_id,
        std::optional<MemoryId> preferred_memory_id,
        void* dst_data,
        size_t num_bytes);
    std::shared_ptr<BlockHeader> read_block_header(BlockId block_id);

    void submit_barrier(EventId id);
    void submit_command(EventId id, Command command, EventList dependencies);
    bool query_event(EventId id, std::chrono::time_point<std::chrono::system_clock> deadline = {});
    void wakeup(std::shared_ptr<Job> job, bool allow_progress = false);
    void shutdown();
    bool is_shutdown();

  private:
    bool make_progress_impl();

    void start_job(std::shared_ptr<Scheduler::Node> node);
    void poll_job(Job& job);
    void stop_job(Job& job);

    std::shared_ptr<SharedJobQueue> m_shared_poll_queue;

    std::mutex m_lock;
    std::condition_variable m_job_completion;
    JobQueue m_local_poll_queue;
    Scheduler m_scheduler;
    WorkerState m_state;
    bool m_shutdown = false;
};

}  // namespace kmm