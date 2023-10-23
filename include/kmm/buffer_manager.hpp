#pragma once

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "kmm/types.hpp"

namespace kmm {

struct BufferRecord;

struct BufferManager {
    BufferId create(BufferLayout layout, MemoryId home);
    void increment_refcount(BufferId id, uint64_t count);
    bool decrement_refcount(BufferId id, uint64_t count);
    void
    update_access(BufferId id, TaskId accessor, AccessMode mode, std::vector<TaskId>& deps_out);

  private:
    BufferId next_buffer_id = 1;
    std::unordered_map<BufferId, std::unique_ptr<BufferRecord>> buffers;
};

}  // namespace kmm