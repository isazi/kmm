#pragma once

#include <chrono>
#include <memory>

#include "cuda_stream_manager.hpp"
#include "memory_manager.hpp"
#include "operation.hpp"

#include "kmm/core/copy_specification.hpp"
#include "kmm/core/device_info.hpp"
#include "kmm/core/task.hpp"
#include "kmm/utils/macros.hpp"
#include "kmm/utils/small_vector.hpp"

namespace kmm {

struct BufferRequirement {
    MemoryId memory_id;
    BufferId buffer_id;
    bool is_write;
    std::optional<EventId> dependency;
};

class Scheduler {
    KMM_NOT_COPYABLE_OR_MOVABLE(Scheduler)

    struct Device {
        CudaContextHandle context;
        std::deque<std::shared_ptr<DeviceOperation>> queue;
        std::vector<std::shared_ptr<DeviceOperation>> active_streams;
    };

  public:
    Scheduler(
        std::shared_ptr<MemoryManager> memory_manager,
        std::shared_ptr<CudaStreamManager> stream_manager);

    BufferId create_buffer(BufferLayout layout);
    void delete_buffer(BufferId id, EventList deps = {});

    EventId insert_host_task(
        std::shared_ptr<HostTask> task,
        std::vector<BufferRequirement> buffers);

    EventId insert_device_task(
        DeviceId device_id,
        std::shared_ptr<DeviceTask> task,
        std::vector<BufferRequirement> buffers);

    EventId insert_copy(
        BufferId src_buffer_id,
        MemoryId src_memory_id,
        std::optional<EventId> src_dependency,
        BufferId dst_buffer_id,
        MemoryId dst_memory_id,
        std::optional<EventId> dst_dependency,
        CopySpecification operation);

    EventId insert_barrier();

    EventId join_events(const EventId* begin, const EventId* end);

    EventId join_events(const EventList& events) {
        return join_events(events.begin(), events.end());
    }

    std::vector<DeviceInfo> collect_device_info();
    bool is_ready(EventId event_id) const;
    bool wait_until_ready(EventId event_id, std::chrono::system_clock::time_point deadline = {});
    bool wait_until_idle(std::chrono::system_clock::time_point deadline = {});
    void make_progress();

  private:
    void make_progress_device(DeviceId device_id);
    std::optional<std::shared_ptr<Operation>> find_event(std::optional<EventId> event_id) const;

    template<typename T, typename... Args>
    std::shared_ptr<T> create_event(Args&&... args) {
        auto id = m_next_event_id;
        m_next_event_id = EventId(m_next_event_id.get() + 1);
        auto op = std::make_shared<T>(id, std::forward<Args>(args)...);

        m_events.emplace(id, op);
        return op;
    }

  public:
    EventId m_next_event_id;
    std::shared_ptr<MemoryManager> m_memory;
    std::shared_ptr<CudaStreamManager> m_streams;
    std::vector<std::shared_ptr<Operation>> m_active_events;
    std::unordered_map<EventId, std::shared_ptr<Operation>> m_events;

    std::vector<Device> m_devices;
};

}  // namespace kmm