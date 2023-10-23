//
// Created by stijn on 10/23/23.
//

#include "kmm/buffer_manager.hpp"

#include <deque>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "kmm/manager.hpp"
#include "kmm/task_graph.hpp"

namespace kmm {

struct BufferRecord {
    BufferRecord(BufferId id, BufferLayout layout, MemoryId home) :
        id_(id),
        layout_(layout),
        home_(home) {}

    BufferRecord(const BufferRecord&) = delete;

    void increment_ref_counter(uint64_t count) {
        if (ref_count_ == 0 || ref_count_ >= std::numeric_limits<uint64_t>::max() - count) {
            throw std::runtime_error("invalid buffer ref count");
        }

        ref_count_ += count;
    }

    bool decrement_ref_counter(uint64_t count) {
        if (ref_count_ < count) {
            throw std::runtime_error("invalid buffer ref count");
        }

        ref_count_ -= count;
        return ref_count_ > 0;
    }

    void update_access(TaskId accessor, AccessMode mode, std::vector<TaskId>& deps_out) {
        switch (mode) {
            case AccessMode::Read:
                deps_out.insert(deps_out.end(), last_writes.begin(), last_writes.end());
                last_readers.push_back(accessor);
                break;
            case AccessMode::Write:
                deps_out.insert(deps_out.end(), last_writes.begin(), last_writes.end());
                deps_out.insert(deps_out.end(), last_readers.begin(), last_readers.end());

                last_readers.clear();
                last_writes = {accessor};
                break;
        }
    }

  private:
    BufferId id_;
    BufferLayout layout_;
    MemoryId home_;
    uint64_t ref_count_ = 1;
    std::vector<TaskId> last_readers;
    std::vector<TaskId> last_writes;
};

BufferId BufferManager::create(BufferLayout layout, MemoryId home) {
    BufferId id = next_buffer_id;
    auto state = std::make_unique<BufferRecord>(id, layout, home);
    buffers.emplace(id, std::move(state));
    return id;
}

void BufferManager::increment_refcount(BufferId id, uint64_t count) {
    buffers.at(id)->increment_ref_counter(count);
}

bool BufferManager::decrement_refcount(BufferId id, uint64_t count) {
    auto it = buffers.find(id);
    if (it == buffers.end()) {
        throw std::runtime_error("invalid buffer id");
    }

    auto& state = it->second;
    if (state->decrement_ref_counter(count)) {
        return false;
    }

    buffers.erase(it);
    return true;
}

void BufferManager::update_access(
    BufferId id,
    TaskId accessor,
    AccessMode mode,
    std::vector<TaskId>& deps_out) {
    buffers.at(id)->update_access(accessor, mode, deps_out);
}
}  // namespace kmm