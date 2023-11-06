#pragma once

#include <mutex>

#include "kmm/dag_builder.hpp"
#include "kmm/memory_manager.hpp"
#include "kmm/scheduler.hpp"
#include "kmm/scheduler_thread.hpp"

namespace kmm {

class RuntimeImpl {
  public:
    RuntimeImpl(std::vector<std::shared_ptr<Executor>> executors, std::shared_ptr<Memory> memory);
    ~RuntimeImpl();

    BufferId create_buffer(const BufferLayout&) const;
    void delete_buffer(BufferId) const;

    OperationId submit_task(
        std::shared_ptr<Task> task,
        TaskRequirements buffers,
        std::vector<OperationId> dependencies) const;

    OperationId submit_barrier() const;
    OperationId submit_buffer_barrier(BufferId) const;
    OperationId submit_promise(OperationId, std::promise<void>) const;

  private:
    mutable std::mutex m_mutex;
    mutable DAGBuilder m_dag_builder;
    std::shared_ptr<ObjectManager> m_object_manager;
    std::shared_ptr<Scheduler> m_scheduler;
    SchedulerThread m_thread;
};

}  // namespace kmm