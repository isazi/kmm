#include "kmm/buffer_manager.hpp"

namespace kmm {

VirtualBufferId BufferManager::create_buffer(const BufferDescription& spec) {
    VirtualBufferId id = m_next_buffer_id;
    m_next_buffer_id = id + 1;

    auto record = std::make_unique<Record>(Record {
        .id = id,
        .name = spec.name,
        .num_bytes = spec.num_bytes,
        .alignment = spec.alignment,
        .refcount = 1,
        .last_writers = {},
        .last_readers = {},
    });

    m_buffers.insert({id, std::move(record)});
    return id;
}

void BufferManager::increment_buffer_references(VirtualBufferId id, uint64_t count) {
    auto& record = m_buffers.at(id);
    if (record->refcount == 0 || record->refcount > std::numeric_limits<uint64_t>::max() - count) {
        throw std::runtime_error("invalid buffer refcount");
    }

    record->refcount += count;
}

bool BufferManager::decrement_buffer_references(VirtualBufferId id, uint64_t count) {
    auto& record = m_buffers.at(id);
    if (record->refcount < count) {
        throw std::runtime_error("invalid buffer refcount");
    }

    record->refcount -= count;

    if (record->refcount == 0) {
        m_buffers.erase(id);
        return true;
    } else {
        return false;
    }
}

void BufferManager::update_buffer_access(
    VirtualBufferId id,
    TaskId task_id,
    AccessMode mode,
    std::vector<TaskId>& deps_out) {
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
}
}  // namespace kmm