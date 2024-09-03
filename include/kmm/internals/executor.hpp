#pragma once

#include "cuda_stream_manager.hpp"
#include "memory_manager.hpp"
#include "scheduler.hpp"

#include "kmm/core/cuda_device.hpp"
#include "kmm/utils/poll.hpp"

namespace kmm {

class Executor;
class Operation;

class Executor {
    KMM_NOT_COPYABLE_OR_MOVABLE(Executor)
  public:
    Executor(
        std::shared_ptr<CudaStreamManager> streams,
        std::shared_ptr<MemoryManager> memory,
        std::shared_ptr<Scheduler> scheduler);
    ~Executor();

    void make_progress();
    bool is_idle() const;

    void submit_host_task(
        std::shared_ptr<TaskNode> job,
        std::shared_ptr<HostTask> task,
        const std::vector<BufferRequirement>& buffers,
        CudaEventSet dependencies = {});

    void submit_device_task(
        std::shared_ptr<TaskNode> job,
        DeviceId device_id,
        std::shared_ptr<DeviceTask> task,
        const std::vector<BufferRequirement>& buffers,
        CudaEventSet dependencies = {});

    void submit_prefetch(
        std::shared_ptr<TaskNode> job,
        BufferId buffer_id,
        MemoryId memory_id,
        CudaEventSet dependencies = {});

    void submit_copy(
        std::shared_ptr<TaskNode> job,
        BufferId src_id,
        MemoryId src_memory,
        BufferId dst_id,
        MemoryId dst_memory,
        CopyDescription spec,
        CudaEventSet dependencies = {});

    friend class Operation;
    friend class HostOperation;
    friend class DeviceOperation;

  private:
    std::shared_ptr<CudaStreamManager> m_streams;
    std::shared_ptr<MemoryManager> m_memory;
    std::shared_ptr<Scheduler> m_scheduler;
    std::vector<std::unique_ptr<Operation>> m_operations;
    std::vector<std::pair<CudaStream, std::unique_ptr<CudaDevice>>> m_devices;
};

}  // namespace kmm