#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "kmm/core/buffer.hpp"
#include "kmm/core/copy_description.hpp"
#include "kmm/internals/commands.hpp"

namespace kmm {

struct Event {
    EventId id;
    Command command;
    EventList dependencies;
};

class TaskGraph {
    KMM_NOT_COPYABLE_OR_MOVABLE(TaskGraph)

  public:
    TaskGraph() = default;

    BufferId create_buffer(BufferLayout layout);

    EventId delete_buffer(BufferId id, EventList deps = {});

    EventId join_events(EventList deps);

    EventId insert_copy(
        BufferId src_buffer,
        MemoryId src_memory,
        BufferId dst_buffer,
        MemoryId dst_memory,
        CopyDescription spec,
        EventList deps = {});

    EventId insert_prefetch(BufferId buffer_id, MemoryId memory_id, EventList deps = {});

    EventId insert_host_task(
        std::shared_ptr<HostTask> task,
        std::vector<BufferRequirement> buffers,
        EventList deps = {});

    EventId insert_device_task(
        DeviceId device_id,
        std::shared_ptr<DeviceTask> task,
        std::vector<BufferRequirement> buffers,
        EventList deps = {});

    EventId insert_barrier();

    EventId shutdown();

    std::vector<Event> flush();

  private:
    EventId insert_event(Command command, EventList deps = {});

    uint64_t m_next_event_id = 1;
    EventList m_events_since_last_barrier;

    struct BufferMeta {
        EventId creation;
        EventList accesses;
    };

    uint64_t m_next_buffer_id = 1;
    std::unordered_map<BufferId, BufferMeta> m_buffers;
    std::vector<Event> m_events;
};

}  // namespace kmm