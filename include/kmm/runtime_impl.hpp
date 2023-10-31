#pragma once

#include <mutex>

#include "kmm/dag_builder.hpp"
#include "kmm/memory_manager.hpp"
#include "kmm/scheduler.hpp"

namespace kmm {

class RuntimeImpl {
  public:
    BufferId create_buffer(const BufferLayout&) const;
    void delete_buffer(BufferId) const;

    OperationId submit_task(
        DeviceId device_id,
        std::shared_ptr<Task> task,
        std::vector<VirtualBufferRequirement> buffers,
        std::vector<OperationId> dependencies) const;

    OperationId submit_barrier() const;
    OperationId submit_buffer_barrier(BufferId) const;
    OperationId submit_promise(OperationId, std::promise<void>) const;

  private:
    mutable std::mutex m_mutex;
    mutable DAGBuilder m_dag_builder;
    std::shared_ptr<Scheduler> m_scheduler;
    std::shared_ptr<ObjectManager> m_object_manager;
};

}  // namespace kmm