#pragma once

#include <mutex>

#include "kmm/buffer_manager.hpp"
#include "kmm/memory_manager.hpp"
#include "kmm/worker.hpp"

namespace kmm {

class RuntimeImpl {
  public:
    VirtualBufferId create_buffer(const BufferDescription&) const;
    void increment_buffer_references(VirtualBufferId, uint64_t count = 1) const;
    void decrement_buffer_references(VirtualBufferId, uint64_t count = 1) const;

    JobId submit_task(
        DeviceId device_id,
        std::shared_ptr<Task> task,
        std::vector<VirtualBufferRequirement> buffers,
        std::vector<JobId> dependencies) const;

  private:
    mutable std::mutex m_mutex;
    mutable BufferManager m_buffer_manager;
    mutable JobId m_next_event_id = JobId(1);
    mutable std::shared_ptr<Worker> m_worker;
};

}  // namespace kmm