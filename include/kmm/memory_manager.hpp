#pragma once

#include <memory>
#include <vector>

#include "kmm/types.hpp"

namespace kmm {

struct BufferState;
struct BufferRequest;

struct MemoryManager {
    void create_buffer(BufferId id, BufferLayout layout);
    void delete_buffer(BufferId id);
    std::shared_ptr<BufferRequest> request_access(BufferId buffer_id, AccessMode mode);
    void release_access(std::shared_ptr<BufferRequest> request);

  private:
    std::unordered_map<BufferId, std::shared_ptr<BufferRecord>> buffers;
    std::vector<std::shared_ptr<Memory>> memories;
};

}  // namespace kmm