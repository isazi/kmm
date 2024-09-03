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

    m_buffers.emplace(buffer_id, BufferMeta {.creation = event_id, .accesses = {event_id}});

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
    deps.push_back(m_buffers.at(src_buffer).creation);
    deps.push_back(m_buffers.at(dst_buffer).creation);

    auto event_id = insert_event(
        CommandCopy {
            src_buffer,  //
            src_memory,
            dst_buffer,
            dst_memory,
            spec},
        std::move(deps));

    m_buffers.at(src_buffer).accesses.push_back(event_id);
    m_buffers.at(dst_buffer).accesses.push_back(event_id);

    return event_id;
}

EventId TaskGraph::insert_prefetch(BufferId buffer_id, MemoryId memory_id, EventList deps) {
    auto event_id = insert_event(
        CommandPrefetch {
            .buffer_id = buffer_id,  //
            .memory_id = memory_id},
        std::move(deps));

    m_buffers.at(buffer_id).accesses.push_back(event_id);
    return event_id;
}

EventId TaskGraph::insert_host_task(
    std::shared_ptr<HostTask> task,
    std::vector<BufferRequirement> buffers,
    EventList deps) {
    for (const auto& buffer : buffers) {
        deps.push_back(m_buffers.at(buffer.buffer_id).creation);
    }

    auto event_id = insert_event(
        CommandExecuteHost {
            .task = std::move(task),  //
            .buffers = buffers},
        std::move(deps));

    for (const auto& buffer : buffers) {
        m_buffers.at(buffer.buffer_id).accesses.push_back(event_id);
    }

    return event_id;
}

EventId TaskGraph::insert_device_task(
    DeviceId device_id,
    std::shared_ptr<DeviceTask> task,
    std::vector<BufferRequirement> buffers,
    EventList deps) {
    for (const auto& buffer : buffers) {
        deps.push_back(m_buffers.at(buffer.buffer_id).creation);
    }

    auto event_id = insert_event(
        CommandExecuteDevice {.device_id = device_id, .task = std::move(task), .buffers = buffers},
        std::move(deps));

    for (const auto& buffer : buffers) {
        m_buffers.at(buffer.buffer_id).accesses.push_back(event_id);
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

}  // namespace kmm