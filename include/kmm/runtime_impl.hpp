#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "kmm/device.hpp"
#include "kmm/event_list.hpp"
#include "kmm/identifiers.hpp"
#include "kmm/worker/runner.hpp"

namespace kmm {

class Worker;

class RuntimeImpl: public std::enable_shared_from_this<RuntimeImpl> {
  public:
    RuntimeImpl(std::vector<std::shared_ptr<DeviceHandle>> devices, std::unique_ptr<Memory> memory);
    ~RuntimeImpl();

    BlockId create_block(
        MemoryId memory_id,
        std::unique_ptr<BlockHeader> header,
        const void* src_data,
        size_t num_bytes) const;
    std::shared_ptr<BlockHeader> read_block_header(BlockId block_id) const;
    std::shared_ptr<BlockHeader> read_block(BlockId block_id, void* dst_data, size_t num_bytes)
        const;
    EventId delete_block(BlockId block_id, EventList deps = {}) const;

    EventId join_events(EventList deps) const;

    EventId submit_barrier() const;
    EventId submit_task(std::shared_ptr<Task> task, TaskRequirements reqs) const;
    EventId submit_block_barrier(BlockId block_id) const;
    EventId submit_block_prefetch(BlockId block_id, MemoryId memory_id, EventList deps = {}) const;

    bool query_event(EventId id, std::chrono::time_point<std::chrono::system_clock> deadline = {})
        const;

    size_t num_devices() const;
    const DeviceInfo& device_info(DeviceId id) const;

  private:
    std::vector<std::unique_ptr<DeviceInfo>> m_devices;
    std::shared_ptr<Worker> m_worker;
    WorkerRunner m_thread;

    mutable std::mutex m_mutex;
    mutable uint64_t m_next_event = 1;
    mutable std::unordered_map<BlockId, EventList> m_block_accesses;
};

}  // namespace kmm