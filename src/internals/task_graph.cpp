#include <algorithm>

#include "spdlog/spdlog.h"

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
    const std::vector<BufferRequirement>& buffers,
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

EventId TaskGraph::insert_reduction(
    ReductionOp op,
    BufferId final_buffer_id,
    MemoryId final_memory_id,
    DataType dtype,
    size_t num_elements,
    std::vector<ReductionInput> inputs) {
    auto layout = BufferLayout {
        .size_in_bytes = dtype.size_in_bytes() * num_elements,
        .alignment = dtype.alignment(),
        .fill_pattern = reduction_identity_value(dtype, op)};

    std::vector<ReductionInput> output_per_device;

    for (const auto& input : inputs) {
        MemoryId local_id = input.memory_id;
        ReductionInput* output = nullptr;

        for (size_t i = 0; i < output_per_device.size(); i++) {
            if (output_per_device[i].memory_id == local_id) {
                output = &output_per_device[i];
            }
        }

        if (output == nullptr) {
            auto temp_buffer_id = BufferId(m_next_buffer_id);
            m_next_buffer_id++;

            auto temp_creation_id =
                insert_event(CommandBufferCreate {.id = temp_buffer_id, .layout = layout});

            output_per_device.push_back(ReductionInput {
                .buffer_id = temp_buffer_id,
                .memory_id = local_id,
                .dependencies = {temp_creation_id},
                .num_inputs_per_output = 1});

            output = &output_per_device.back();
        }

        auto deps = input.dependencies;
        deps.push_back(output->dependencies[0]);

        auto event_id = insert_event(
            CommandReduction {
                .src_buffer = input.buffer_id,
                .dst_buffer = output->buffer_id,
                .memory_id = local_id,
                .reduction =
                    Reduction {
                        .operation = op,
                        .data_type = dtype,
                        .num_outputs = num_elements,
                        .num_inputs_per_output = input.num_inputs_per_output,
                    }},
            std::move(deps));

        output->dependencies.push_back(event_id);

        m_buffer_accesses.push_back(BufferAccess {
            .buffer_id = input.buffer_id,
            .access_mode = AccessMode::Read,
            .event_id = event_id});
    }

    for (size_t j = 0; j < output_per_device.size(); j++) {
        auto& partial_output = output_per_device[j];
        auto deps = partial_output.dependencies;
        access_buffer(final_buffer_id, AccessMode::Exclusive, deps);

        auto event_id = insert_event(
            CommandReduction {
                .src_buffer = partial_output.buffer_id,
                .dst_buffer = final_buffer_id,
                .memory_id = final_memory_id,
                .reduction =
                    Reduction {
                        .operation = op,
                        .data_type = dtype,
                        .num_outputs = num_elements,
                        .num_inputs_per_output = 1}},
            std::move(deps));

        m_buffer_accesses.push_back(BufferAccess {
            .buffer_id = final_buffer_id,
            .access_mode = AccessMode::Exclusive,
            .event_id = event_id});

        insert_event(CommandBufferDelete {.id = partial_output.buffer_id}, {event_id});
    }
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

    auto current = m_buffer_accesses.begin();

    while (current != m_buffer_accesses.end()) {
        EventList write_events;
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
            meta.last_writes = std::move(write_events);
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

    if (mode != AccessMode::Read) {
        deps_out.insert_all(meta.accesses);
    }
}

}  // namespace kmm