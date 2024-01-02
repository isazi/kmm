#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "kmm/executor.hpp"
#include "kmm/types.hpp"
#include "kmm/worker/command.hpp"
#include "kmm/worker/memory_manager.hpp"
#include "kmm/worker/worker.hpp"

namespace kmm {

class ExecuteJob: public Job {
  public:
    ExecuteJob(EventId id, ExecuteCommand command) :
        Job(id),
        m_id(id),
        m_device_id(command.executor_id),
        m_task(std::move(command.task)),
        m_inputs(std::move(command.inputs)),
        m_outputs(std::move(command.outputs)) {}

    PollResult poll(WorkerState&) final;

    //  private:
    enum class Status { Created, Staging, Running, Done };
    Status m_status = Status::Created;

    EventId m_id;
    ExecutorId m_device_id;
    std::shared_ptr<Task> m_task;
    std::vector<TaskInput> m_inputs;
    std::vector<TaskOutput> m_outputs;
    std::vector<std::optional<BufferId>> m_output_buffers;
    std::vector<MemoryRequest> m_memory_requests;

    struct TaskResult;
    std::shared_ptr<TaskResult> m_result = nullptr;
};

class DeleteJob: public Job {
  public:
    DeleteJob(EventId id, BlockDeleteCommand cmd) : Job(id), m_block_id(cmd.id) {}
    PollResult poll(WorkerState&) final;

  private:
    std::optional<BlockId> m_block_id;
    std::optional<BufferId> m_buffer_id;
};

class PrefetchJob: public Job {
  public:
    PrefetchJob(EventId id, BlockPrefetchCommand cmd) :
        Job(id),
        m_memory_id(cmd.memory_id),
        m_block_id(cmd.block_id) {}

    PollResult poll(WorkerState&) final;

  private:
    enum class Status { Created, Active, Done };
    Status m_status;
    MemoryRequest m_memory_request = nullptr;
    MemoryId m_memory_id;
    BlockId m_block_id;
};

class EmptyJob: public Job {
  public:
    EmptyJob(EventId id) : Job(id) {}
    PollResult poll(WorkerState&) final;
};

std::shared_ptr<Job> build_job_for_command(EventId id, Command command);

}  // namespace kmm