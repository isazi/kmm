#include <algorithm>

#include "spdlog/spdlog.h"

#include "kmm/internals/task_graph.hpp"

namespace kmm {

BufferId TaskGraph::create_buffer(BufferLayout layout) {
    auto [buffer_id, event_id] = insert_create_buffer_event(std::move(layout));
    m_tentative_buffers.emplace(buffer_id, BufferMeta(event_id));

    return buffer_id;
}

EventId TaskGraph::delete_buffer(BufferId id, EventList deps) {
    // Find the buffer
    auto& meta = find_buffer(id);

    // Erase the buffer
    m_tentative_deletions.push_back(id);

    // Get the list of accesses + user-provided dependencies
    deps.push_back(meta.creation);
    deps.insert_all(meta.accesses);

    return insert_delete_buffer_event(id, std::move(deps));
}

const EventList& TaskGraph::extract_buffer_dependencies(BufferId id) {
    return find_buffer(id).accesses;
}

std::pair<BufferId, EventId> TaskGraph::insert_create_buffer_event(BufferLayout layout) {
    auto buffer_id = BufferId(m_next_buffer_id);
    m_next_buffer_id++;

    auto event_id = insert_event(CommandBufferCreate {
        .id = buffer_id,
        .layout = layout,
    });

    return {buffer_id, event_id};
}

EventId TaskGraph::insert_delete_buffer_event(BufferId id, EventList deps) {
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
    CopyDef spec,
    EventList deps
) {
    pre_access_buffer(src_buffer, AccessMode::Read, src_memory, deps);
    pre_access_buffer(dst_buffer, AccessMode::ReadWrite, dst_memory, deps);

    auto event_id = insert_event(
        CommandCopy {
            src_buffer,  //
            src_memory,
            dst_buffer,
            dst_memory,
            spec},
        std::move(deps)
    );

    post_access_buffer(src_buffer, AccessMode::Read, src_memory, event_id);
    post_access_buffer(dst_buffer, AccessMode::ReadWrite, dst_memory, event_id);

    return event_id;
}

EventId TaskGraph::insert_prefetch(BufferId buffer_id, MemoryId memory_id, EventList deps) {
    pre_access_buffer(buffer_id, AccessMode::Read, memory_id, deps);

    auto event_id = insert_event(
        CommandPrefetch {
            .buffer_id = buffer_id,  //
            .memory_id = memory_id},
        std::move(deps)
    );

    post_access_buffer(buffer_id, AccessMode::Read, memory_id, event_id);
    return event_id;
}

EventId TaskGraph::insert_task(
    ProcessorId processor_id,
    std::shared_ptr<Task> task,
    const std::vector<BufferRequirement>& buffers,
    EventList deps
) {
    for (const auto& buffer : buffers) {
        pre_access_buffer(buffer.buffer_id, buffer.access_mode, buffer.memory_id, deps);
    }

    auto event_id = insert_event(
        CommandExecute {.processor_id = processor_id, .task = std::move(task), .buffers = buffers},
        std::move(deps)
    );

    for (const auto& buffer : buffers) {
        post_access_buffer(buffer.buffer_id, buffer.access_mode, buffer.memory_id, event_id);
    }

    return event_id;
}

EventId TaskGraph::insert_multilevel_reduction(
    BufferId final_buffer_id,
    MemoryId final_memory_id,
    Reduction reduction,
    std::vector<ReductionInput> inputs
) {
    auto op = reduction.operation;
    auto dtype = reduction.data_type;
    auto num_elements = reduction.num_outputs;

    if (std::all_of(inputs.begin(), inputs.end(), [&](const auto& a) {
            return a.memory_id == final_memory_id;
        })) {
        return insert_local_reduction(final_memory_id, final_buffer_id, reduction, inputs);
    }

    std::stable_sort(inputs.begin(), inputs.end(), [&](const auto& a, const auto& b) {
        return a.memory_id < b.memory_id;
    });

    auto temporary_layout = BufferLayout::for_type(dtype).repeat(num_elements);
    auto temporary_buffers = std::vector<BufferId> {};
    std::vector<ReductionInput> result_per_device;
    size_t cursor = 0;

    while (cursor < inputs.size()) {
        auto memory_id = inputs[cursor].memory_id;
        size_t begin = cursor;

        while (cursor < inputs.size() && memory_id == inputs[cursor].memory_id) {
            cursor++;
        }

        size_t length = cursor - begin;

        // Special case: if there is only one buffer having one input, then we do not need to
        // create a local scratch buffer.
        if (length == 1 && inputs[begin].num_inputs_per_output == 1) {
            result_per_device.push_back(inputs[begin]);
            continue;
        }

        auto local_inputs = std::vector<ReductionInput> {&inputs[begin], &inputs[cursor]};

        auto local_buffer_id = create_buffer(temporary_layout);
        auto event_id = insert_local_reduction(memory_id, local_buffer_id, reduction, local_inputs);

        temporary_buffers.push_back(local_buffer_id);
        result_per_device.push_back(ReductionInput {
            .buffer_id = local_buffer_id,
            .memory_id = memory_id,
            .dependencies = {event_id}});
    }

    auto event_id =
        insert_local_reduction(final_memory_id, final_buffer_id, reduction, result_per_device);

    for (auto& buffer_id : temporary_buffers) {
        delete_buffer(buffer_id);
    }

    return event_id;
}

EventId TaskGraph::insert_local_reduction(
    MemoryId memory_id,
    BufferId buffer_id,
    Reduction reduction,
    std::vector<ReductionInput> inputs
) {
    auto dtype = reduction.data_type;
    auto op = reduction.operation;
    auto num_elements = reduction.num_outputs;

    if (inputs.size() == 1) {
        const auto& input = inputs[0];
        auto deps = input.dependencies;

        pre_access_buffer(input.buffer_id, AccessMode::Read, input.memory_id, deps);
        pre_access_buffer(buffer_id, AccessMode::ReadWrite, memory_id, deps);

        EventId event_id = insert_reduction_event(
            input.buffer_id,
            input.memory_id,
            buffer_id,
            memory_id,
            ReductionDef {
                .operation = op,
                .data_type = dtype,
                .num_outputs = num_elements,
                .num_inputs_per_output = input.num_inputs_per_output},
            std::move(deps)
        );

        post_access_buffer(input.buffer_id, AccessMode::Read, input.memory_id, event_id);
        post_access_buffer(buffer_id, AccessMode::ReadWrite, memory_id, event_id);

        return event_id;
    }

    auto scratch_layout = BufferLayout::for_type(dtype).repeat(num_elements).repeat(inputs.size());
    auto [scratch_id, scratch_creation] = insert_create_buffer_event(scratch_layout);
    auto scratch_deps = EventList {};

    for (size_t i = 0; i < inputs.size(); i++) {
        const auto& input = inputs[i];
        auto deps = input.dependencies;

        pre_access_buffer(input.buffer_id, AccessMode::Read, input.memory_id, deps);
        deps.push_back(scratch_creation);

        EventId event_id = insert_reduction_event(
            input.buffer_id,
            input.memory_id,
            scratch_id,
            memory_id,
            ReductionDef {
                .operation = op,
                .data_type = dtype,
                .num_outputs = num_elements,
                .num_inputs_per_output = input.num_inputs_per_output,
                .src_offset_elements = 0,
                .dst_offset_elements = i * num_elements},
            std::move(deps)
        );

        post_access_buffer(input.buffer_id, AccessMode::Read, input.memory_id, event_id);
        scratch_deps.push_back(event_id);
    }

    pre_access_buffer(buffer_id, AccessMode::ReadWrite, memory_id, scratch_deps);

    auto event_id = insert_reduction_event(
        scratch_id,
        memory_id,
        buffer_id,
        memory_id,
        ReductionDef {
            .operation = op,
            .data_type = dtype,
            .num_outputs = num_elements,
            .num_inputs_per_output = inputs.size(),
        },
        std::move(scratch_deps)
    );

    post_access_buffer(buffer_id, AccessMode::ReadWrite, memory_id, event_id);
    insert_delete_buffer_event(scratch_id, {event_id});

    return event_id;
}

EventId TaskGraph::insert_reduction_event(
    BufferId src_buffer,
    MemoryId src_memory_id,
    BufferId dst_buffer,
    MemoryId dst_memory_id,
    ReductionDef reduction,
    EventList deps
) {
    if (reduction.num_inputs_per_output == 1) {
        auto dtype = reduction.data_type;
        auto src_offset = reduction.src_offset_elements;
        auto dst_offset = reduction.dst_offset_elements;
        size_t num_elements = reduction.num_outputs;

        auto copy = CopyDef(dtype.size_in_bytes());
        copy.add_dimension(num_elements, src_offset, dst_offset, 1, 1);

        return insert_event(
            CommandCopy {
                .src_buffer = src_buffer,
                .src_memory = src_memory_id,
                .dst_buffer = dst_buffer,
                .dst_memory = dst_memory_id,
                .definition = copy},
            std::move(deps)
        );
    } else {
        return insert_event(
            CommandReduction {
                .src_buffer = src_buffer,
                .dst_buffer = dst_buffer,
                .memory_id = dst_memory_id,
                .definition = reduction},
            std::move(deps)
        );
    }
}

EventId TaskGraph::insert_barrier() {
    EventList deps = std::move(m_events_since_last_barrier);
    return insert_event(CommandEmpty {}, std::move(deps));
}

EventId TaskGraph::shutdown() {
    EventList deps = {insert_barrier()};

    for (auto& [id, buffer] : m_persistent_buffers) {
        m_tentative_deletions.push_back(id);

        auto event_id = insert_event(CommandBufferDelete {id}, buffer.accesses);
        deps.push_back(event_id);
    }

    return join_events(deps);
}

void TaskGraph::rollback() {
    m_tentative_deletions.clear();
    m_tentative_buffers.clear();
}

void TaskGraph::commit() {
    for (auto& [id, meta] : m_tentative_buffers) {
        m_persistent_buffers.insert_or_assign(id, std::move(meta));
    }

    for (auto& id : m_tentative_deletions) {
        m_persistent_buffers.erase(id);
    }

    m_tentative_deletions.clear();
    m_tentative_buffers.clear();
}

std::vector<Event> TaskGraph::flush() {
    rollback();
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

TaskGraph::BufferMeta& TaskGraph::find_buffer(BufferId id) {
    if (auto it = m_tentative_buffers.find(id); KMM_LIKELY(it != m_tentative_buffers.end())) {
        return it->second;
    }

    if (auto it = m_persistent_buffers.find(id); it != m_persistent_buffers.end()) {
        auto [m, success] = m_tentative_buffers.emplace(id, it->second);

        // The item must be inserted, since we did not find it previously.
        KMM_ASSERT(success);

        return m->second;
    }

    throw std::runtime_error(fmt::format("buffer with identifier {} was not found", id));
}

void TaskGraph::pre_access_buffer(
    BufferId buffer_id,
    AccessMode mode,
    MemoryId memory_id,
    EventList& deps_out
) {
    auto& meta = find_buffer(buffer_id);
    deps_out.push_back(meta.creation);

    if (mode != AccessMode::Read) {
        deps_out.insert_all(meta.accesses);
    } else {
        deps_out.push_back(meta.last_write);
    }
}

void TaskGraph::post_access_buffer(
    BufferId buffer_id,
    AccessMode mode,
    MemoryId memory_id,
    EventId new_event_id
) {
    auto& meta = find_buffer(buffer_id);

    if (mode != AccessMode::Read) {
        meta.last_write = new_event_id;
        meta.accesses = {new_event_id};
        meta.owner_id = memory_id;
    } else {
        meta.accesses.push_back(new_event_id);
    }
}

}  // namespace kmm