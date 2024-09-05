#include <algorithm>

#include "kmm/internals/task_graph.hpp"

namespace kmm {

BufferId kmm::TaskGraph::create_buffer(BufferLayout layout) {
    auto buffer_id = BufferId(m_next_buffer_id);
    m_next_buffer_id++;

    auto event_id = insert_event(CommandBufferCreate {
        .id = buffer_id,
        .layout = layout,
    });

    m_buffers.emplace(
        buffer_id,
        BufferMeta {.creation = event_id, .last_writes = {event_id}, .accesses = {event_id}});

    return buffer_id;
}

EventId TaskGraph::delete_buffer(BufferId id, EventList deps) {
    // Find the buffer
    auto it = m_buffers.find(id);
    KMM_ASSERT(it != m_buffers.end());

    // Get the list of accesses + user-provided dependencies
    deps.insert_all(it->second.accesses);

    // Erase the buffer
    m_buffers.erase(it);

    return insert_event(CommandBufferDelete {id}, std::move(deps));
}

EventId TaskGraph::join_events(EventList deps) {
    if (deps.size() == 0) {
        return EventId(0);
    }

    // Check if all dependencies are the same event ID
    if (std::equal(deps.begin() + 1, deps.end(), deps.begin())) {
        return deps[0];
    }

    return insert_event(CommandEmpty {}, std::move(deps));
}

EventId TaskGraph::insert_copy(
    BufferId src_buffer,
    MemoryId src_memory,
    BufferId dst_buffer,
    MemoryId dst_memory,
    CopyDescription spec,
    EventList deps) {
    access_buffer(src_buffer, AccessMode::Read, deps);
    access_buffer(dst_buffer, AccessMode::ReadWrite, deps);

    auto event_id = insert_event(
        CommandCopy {
            src_buffer,  //
            src_memory,
            dst_buffer,
            dst_memory,
            spec},
        std::move(deps));

    m_buffer_accesses.push_back({src_buffer, AccessMode::Read, event_id});
    m_buffer_accesses.push_back({dst_buffer, AccessMode::ReadWrite, event_id});

    return event_id;
}

EventId TaskGraph::insert_prefetch(BufferId buffer_id, MemoryId memory_id, EventList deps) {
    auto event_id = insert_event(
        CommandPrefetch {
            .buffer_id = buffer_id,  //
            .memory_id = memory_id},
        std::move(deps));

    m_buffer_accesses.push_back({buffer_id, AccessMode::Read, event_id});
    return event_id;
}

EventId TaskGraph::insert_task(
    ProcessorId processor_id,
    std::shared_ptr<Task> task,
    std::vector<BufferRequirement> buffers,
    EventList deps) {
    for (const auto& buffer : buffers) {
        access_buffer(buffer.buffer_id, buffer.access_mode, deps);
    }

    auto event_id = insert_event(
        CommandExecute {.processor_id = processor_id, .task = std::move(task), .buffers = buffers},
        std::move(deps));

    for (const auto& buffer : buffers) {
        m_buffer_accesses.push_back({buffer.buffer_id, buffer.access_mode, event_id});
    }

    return event_id;
}

EventId TaskGraph::insert_barrier() {
    EventList deps = std::move(m_events_since_last_barrier);
    return insert_event(CommandEmpty {}, std::move(deps));
}

EventId TaskGraph::shutdown() {
    EventList deps = {insert_barrier()};

    for (auto& [id, buffer] : m_buffers) {
        auto event_id = insert_event(CommandBufferDelete {id}, buffer.accesses);
        deps.push_back(event_id);
    }

    // Delete buffers
    m_buffers.clear();

    return join_events(deps);
}

void TaskGraph::commit() {
    std::stable_sort(
        m_buffer_accesses.begin(),
        m_buffer_accesses.end(),
        [](const auto& a, const auto& b) { return a.buffer_id < b.buffer_id; });

    EventList write_events;
    auto current = m_buffer_accesses.begin();

    while (current != m_buffer_accesses.end()) {
        auto buffer_id = current->buffer_id;
        auto& meta = m_buffers.at(buffer_id);

        do {
            if (current->access_mode != AccessMode::Read) {
                write_events.push_back(current->event_id);
            }

            meta.accesses.push_back(current->event_id);
            current++;
        } while (current != m_buffer_accesses.end() && current->buffer_id == buffer_id);

        if (!write_events.is_empty()) {
            meta.last_writes = write_events;
        }
    }

    m_buffer_accesses.clear();
}

std::vector<Event> TaskGraph::flush() {
    return std::move(m_events);
}

EventId TaskGraph::insert_event(Command command, EventList deps) {
    auto event_id = EventId(m_next_event_id);
    m_next_event_id++;

    if (deps.size() > 1) {
        std::sort(deps.begin(), deps.end());
        auto* mid = std::unique(deps.begin(), deps.end());

        if (mid != deps.end()) {
            deps.resize(mid - deps.begin());
        }
    }

    m_events.push_back({event_id, std::move(command), std::move(deps)});
    m_events_since_last_barrier.push_back(event_id);

    return event_id;
}

void TaskGraph::access_buffer(BufferId buffer_id, AccessMode mode, EventList& deps_out) {
    auto& meta = m_buffers.at(buffer_id);
    deps_out.push_back(meta.creation);
    deps_out.insert_all(meta.last_writes);
}

}  // namespace kmm