#pragma once

#include "kmm/runtime.hpp"
#include "kmm/worker.hpp"

namespace kmm {

class BufferManager {
  public:
    VirtualBufferId create_buffer(const BufferDescription&);
    void increment_buffer_references(VirtualBufferId, uint64_t count);
    bool decrement_buffer_references(VirtualBufferId, uint64_t count);
    BufferRequirement
    update_buffer_access(VirtualBufferId, JobId, AccessMode, std::vector<JobId>& deps_out);

  private:
    struct Record {
        VirtualBufferId virtual_id;
        BufferId physical_id;
        std::string name;
        uint64_t refcount;
        std::vector<JobId> last_writers;
        std::vector<JobId> last_readers;
    };

    VirtualBufferId m_next_buffer_id = VirtualBufferId(1);
    std::unordered_map<VirtualBufferId, std::unique_ptr<Record>> m_buffers;
};

}  // namespace kmm