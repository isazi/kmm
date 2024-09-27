#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "kmm/core/buffer.hpp"
#include "kmm/core/copy_description.hpp"
#include "kmm/internals/commands.hpp"
#include "kmm/utils/macros.hpp"

namespace kmm {

struct Event {
    EventId id;
    Command command;
    EventList dependencies;
};

struct ReductionInput {
    BufferId buffer_id;
    MemoryId memory_id;
    EventList dependencies;
    size_t num_inputs_per_output = 1;
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

    EventId insert_task(
        ProcessorId processor_id,
        std::shared_ptr<Task> task,
        const std::vector<BufferRequirement>& buffers,
        EventList deps = {});

    EventId insert_reduction(
        ReductionOp op,
        BufferId final_buffer_id,
        MemoryId final_memory_id,
        DataType dtype,
        size_t num_outputs,
        std::vector<ReductionInput> inputs);

    EventId insert_barrier();

    EventId shutdown();

    void commit();

    std::vector<Event> flush();

    void access_buffer(BufferId buffer_id, AccessMode mode, EventList& deps_out);

  private:
    EventId insert_event(Command command, EventList deps = {});

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
    std::vector<BufferAccess> m_buffer_accesses;
};

}  // namespace kmm