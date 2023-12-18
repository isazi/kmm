#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "kmm/worker.hpp"

namespace kmm {

class ExecuteJob: public WorkerJob {
  public:
    ExecuteJob(EventId id, CommandExecute command) :
        WorkerJob(id),
        m_device_id(command.device_id),
        m_task(std::move(command.task)),
        m_inputs(std::move(command.inputs)),
        m_outputs(std::move(command.outputs)) {}

    PollResult poll(WorkerState&) final;

  private:
    enum class Status { Created, Staging, Running, Done };
    Status m_status = Status::Created;

    DeviceId m_device_id;
    std::shared_ptr<Task> m_task;
    std::vector<TaskInput> m_inputs;
    std::vector<TaskOutput> m_outputs;
    std::vector<std::optional<BufferId>> m_output_buffers;
    std::vector<MemoryRequest> m_memory_requests;

    struct Result;
    std::shared_ptr<Result> m_result = nullptr;
};

class DeleteJob: public WorkerJob {
  public:
    DeleteJob(EventId id, BlockId block_id) : WorkerJob(id), m_block_id(block_id) {}
    PollResult poll(WorkerState&) final;

  private:
    BlockId m_block_id;
};

class PrefetchJob: public WorkerJob {
  public:
    PrefetchJob(EventId id, DeviceId device_id, BlockId block_id) :
        WorkerJob(id),
        m_device_id(device_id),
        m_block_id(block_id) {}

    PollResult poll(WorkerState&) final;

  private:
    enum class Status { Created, Active, Done };
    Status m_status;
    MemoryRequest m_memory_request = nullptr;
    DeviceId m_device_id;
    BlockId m_block_id;
};

class EmptyJob: public WorkerJob {
  public:
    EmptyJob(EventId id) : WorkerJob(id) {}
    PollResult poll(WorkerState&) final;
};
}  // namespace kmm