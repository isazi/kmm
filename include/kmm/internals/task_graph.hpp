#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "kmm/core/buffer.hpp"
#include "kmm/core/copy_def.hpp"
#include "kmm/core/reduction.hpp"
#include "kmm/internals/commands.hpp"
#include "kmm/utils/macros.hpp"

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

    const EventList& extract_buffer_dependencies(BufferId id) const;

    EventId join_events(EventList deps);

    EventId insert_copy(
        BufferId src_buffer,
        MemoryId src_memory,
        BufferId dst_buffer,
        MemoryId dst_memory,
        CopyDef spec,
        EventList deps = {}
    );

    EventId insert_prefetch(BufferId buffer_id, MemoryId memory_id, EventList deps = {});

    EventId insert_task(
        ProcessorId processor_id,
        std::shared_ptr<Task> task,
        const std::vector<BufferRequirement>& buffers,
        EventList deps = {}
    );

    EventId insert_multilevel_reduction(
        BufferId final_buffer_id,
        MemoryId final_memory_id,
        Reduction reduction,
        std::vector<ReductionInput> inputs
    );

    EventId insert_barrier();

    EventId shutdown();

    void commit();

    std::vector<Event> flush();

  private:
    EventId insert_event(Command command, EventList deps = {});

    std::pair<BufferId, EventId> create_internal_buffer(BufferLayout layout);
    EventId delete_internal_buffer(BufferId id, EventList deps);

    void pre_access_buffer(BufferId buffer_id, AccessMode mode, EventList& deps_out);
    void post_access_buffer(BufferId buffer_id, AccessMode mode, EventId new_event_id);

    EventId insert_local_reduction(
        MemoryId memory_id,
        BufferId buffer_id,
        Reduction reduction,
        const ReductionInput* inputs,
        size_t num_inputs
    );

    EventId insert_reduction_event(
        BufferId src_buffer,
        MemoryId src_memory,
        BufferId dst_buffer,
        MemoryId dst_memory,
        ReductionDef reduction,
        EventList deps
    );

    uint64_t m_next_event_id = 1;
    EventList m_events_since_last_barrier;

    struct BufferMeta {
        EventId creation;
        EventList last_writes;
        EventList accesses;
    };

    struct BufferAccess {
        BufferId buffer_id;
        AccessMode access_mode;
        EventId event_id;
    };

    uint64_t m_next_buffer_id = 1;
    std::unordered_map<BufferId, BufferMeta> m_buffers;
    std::vector<Event> m_events;
};

}  // namespace kmm