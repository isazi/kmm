#pragma once

#include <mutex>

#include "kmm/buffer_manager.hpp"
#include "kmm/memory_manager.hpp"
#include "kmm/task_graph.hpp"

namespace kmm {

class RuntimeImpl {
  public:
    VirtualBufferId create_buffer(const BufferDescription&) const;
    void increment_buffer_references(VirtualBufferId, uint64_t count = 1) const;
    void decrement_buffer_references(VirtualBufferId, uint64_t count = 1) const;

    TaskId submit_task(
        DeviceId device_id,
        std::shared_ptr<Task> task,
        std::vector<BufferRequirement> buffers,
        std::vector<TaskId> dependencies) const;

  private:
    mutable std::mutex m_mutex;
    mutable BufferManager m_buffer_manager;
    mutable TaskId m_next_task_id = TaskId(1);
    mutable TaskGraph m_task_graph;
    mutable MemoryManager m_memory_manager;
};

}  // namespace kmm