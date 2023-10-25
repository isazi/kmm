#pragma once

#include "kmm/runtime.hpp"

namespace kmm {

class BufferManager {
  public:
    VirtualBufferId create_buffer(const BufferDescription&);
    void increment_buffer_references(VirtualBufferId, uint64_t count);
    bool decrement_buffer_references(VirtualBufferId, uint64_t count);
    void update_buffer_access(VirtualBufferId, TaskId, AccessMode, std::vector<TaskId>& deps_out);

  private:
    struct Record {
        VirtualBufferId id;
        std::string name;
        size_t num_bytes;
        size_t alignment;
        uint64_t refcount;
        std::vector<TaskId> last_writers;
        std::vector<TaskId> last_readers;
    };

    VirtualBufferId m_next_buffer_id = VirtualBufferId(1);
    std::unordered_map<VirtualBufferId, std::unique_ptr<Record>> m_buffers;
};

}  // namespace kmm