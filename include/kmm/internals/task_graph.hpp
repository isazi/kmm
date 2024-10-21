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

    const EventList& extract_buffer_dependencies(BufferId id);

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

    EventId insert_local_reduction(
        MemoryId memory_id,
        BufferId buffer_id,
        Reduction reduction,
        std::vector<ReductionInput> inputs
    );

    EventId insert_barrier();

    EventId shutdown();

    void rollback();
    void commit();

    std::vector<Event> flush();

  private:
    EventId insert_event(Command command, EventList deps = {});

    std::pair<BufferId, EventId> insert_create_buffer_event(BufferLayout layout);

    EventId insert_delete_buffer_event(BufferId id, EventList deps);

    EventId insert_reduction_event(
        BufferId src_buffer,
        MemoryId src_memory,
        BufferId dst_buffer,
        MemoryId dst_memory,
        ReductionDef reduction,
        EventList deps
    );

    struct BufferMeta {
        BufferMeta(EventId epoch_event) :
            creation(epoch_event),
            last_write(epoch_event),
            accesses {epoch_event} {}

        MemoryId owner_id = MemoryId::host();
        EventId creation;
        EventId last_write;
        EventList accesses;
    };

    BufferMeta& find_buffer(BufferId id);
    void pre_access_buffer(
        BufferId buffer_id,
        AccessMode mode,
        MemoryId memory_id,
        EventList& deps_out
    );
    void post_access_buffer(
        BufferId buffer_id,
        AccessMode mode,
        MemoryId memory_id,
        EventId new_event_id
    );

    uint64_t m_next_buffer_id = 1;
    uint64_t m_next_event_id = 1;
    EventList m_events_since_last_barrier;
    std::unordered_map<BufferId, BufferMeta> m_persistent_buffers;
    std::unordered_map<BufferId, BufferMeta> m_tentative_buffers;
    std::vector<BufferId> m_tentative_deletions;
    std::vector<Event> m_events;
};

}  // namespace kmm