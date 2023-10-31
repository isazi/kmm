#include "kmm/dag_builder.hpp"

#include "kmm/scheduler.hpp"

namespace kmm {

BufferId DAGBuilder::create_buffer(const BufferLayout& spec) {
    auto id = m_next_buffer_id++;
    auto create_id = m_next_job_id++;

    auto record = std::make_unique<Record>(Record {
        .virtual_id = id,
        .physical_id = PhysicalBufferId(id),
        .name = spec.name,
        .last_writers = {create_id},
        .last_readers = {},
    });

    auto cmd = CommandBufferCreate {
        .id = record->physical_id,
        .description = spec,
    };

    m_commands.emplace_back(create_id, std::move(cmd), std::vector<OperationId> {});
    return id;
}

void DAGBuilder::delete_buffer(BufferId id) {
    auto& record = m_buffers.at(id);

    auto deps = std::vector<OperationId> {};
    deps.insert(deps.end(), record->last_readers.begin(), record->last_readers.end());
    deps.insert(deps.end(), record->last_writers.begin(), record->last_writers.end());

    auto cmd = CommandBufferDelete {record->physical_id};
    m_commands.emplace_back(m_next_job_id++, cmd, deps);
    m_buffers.erase(id);
}

PhysicalBufferId DAGBuilder::update_buffer_access(
    BufferId id,
    OperationId task_id,
    AccessMode mode,
    std::vector<OperationId>& deps_out) {
    auto& record = m_buffers.at(id);

    switch (mode) {
        case AccessMode::Read:
            deps_out.insert(
                deps_out.end(),
                record->last_writers.begin(),
                record->last_writers.end());
            record->last_readers.push_back(task_id);
            break;

        case AccessMode::Write:
            deps_out.insert(
                deps_out.end(),
                record->last_writers.begin(),
                record->last_readers.end());
            deps_out.insert(
                deps_out.end(),
                record->last_readers.begin(),
                record->last_readers.end());

            record->last_readers = {};
            record->last_writers = {task_id};
            break;
    }

    return record->physical_id;
}

OperationId DAGBuilder::submit_task(
    DeviceId device_id,
    std::shared_ptr<Task> task,
    const std::vector<VirtualBufferRequirement>& virtual_buffers,
    std::vector<OperationId> dependencies) {
    auto task_id = m_next_job_id++;
    auto buffers = std::vector<BufferRequirement> {};

    for (const auto& req : virtual_buffers) {
        auto buffer_id = update_buffer_access(
            req.buffer_id,  //
            task_id,
            req.mode,
            dependencies);

        buffers.push_back(BufferRequirement {
            .buffer_id = buffer_id,
            .memory_id = device_id,
            .is_write = req.mode != AccessMode::Read,
        });
    }

    auto cmd = CommandExecute {
        .output_object_id = ObjectId::invalid(),
        .device_id = device_id,
        .task = std::move(task),
        .buffers = std::move(buffers),
    };

    m_commands.emplace_back(task_id, std::move(cmd), std::move(dependencies));
    return task_id;
}

OperationId DAGBuilder::submit_barrier() {
    auto dependencies = std::vector<OperationId> {};
    OperationId task_id = m_next_job_id++;

    for (const auto& [_, buffer] : m_buffers) {
        dependencies.insert(
            dependencies.end(),
            buffer->last_readers.begin(),
            buffer->last_readers.end());

        dependencies.insert(
            dependencies.end(),
            buffer->last_writers.begin(),
            buffer->last_writers.end());

        buffer->last_readers = {};
        buffer->last_writers = {task_id};
    }

    m_commands.emplace_back(task_id, CommandNoop {}, std::move(dependencies));
    return task_id;
}

OperationId DAGBuilder::submit_buffer_barrier(BufferId buffer_id) {
    auto dependencies = std::vector<OperationId> {};
    OperationId task_id = m_next_job_id++;

    auto& buffer = m_buffers.at(buffer_id);
    dependencies.insert(
        dependencies.end(),
        buffer->last_readers.begin(),
        buffer->last_readers.end());

    dependencies.insert(
        dependencies.end(),
        buffer->last_writers.begin(),
        buffer->last_writers.end());

    buffer->last_readers = {};
    buffer->last_writers = {task_id};

    m_commands.emplace_back(task_id, CommandNoop {}, std::move(dependencies));
    return task_id;
}

OperationId DAGBuilder::submit_promise(OperationId op_id, std::promise<void> promise) {
    OperationId task_id = m_next_job_id++;
    std::vector<OperationId> dependencies = {op_id};
    m_commands.emplace_back(
        task_id,
        CommandPromise {.promise = std::move(promise)},
        std::move(dependencies));
    return task_id;
}

std::vector<CommandPacket> DAGBuilder::flush() {
    return std::move(m_commands);
}

void DAGBuilder::flush(Scheduler& scheduler) {
    for (auto& packet : m_commands) {
        scheduler.submit_command(std::move(packet));
    }

    m_commands.clear();
}
}  // namespace kmm