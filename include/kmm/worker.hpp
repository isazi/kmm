#pragma once

#include <deque>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

#include "kmm/memory_manager.hpp"
#include "kmm/types.hpp"

namespace kmm {

class ExecutorContext;
class TaskContext;

class Task {
  public:
    virtual ~Task() = default;
    virtual void execute(ExecutorContext&, TaskContext&) = 0;
};

struct CommandExecute {
    DeviceId device_id;
    std::shared_ptr<Task> task;
};

struct CommandNoop {};

struct CommandBufferCreate {
    BufferId id;
    BufferDescription description;
};

struct CommandBufferDelete {
    BufferId id;
};

using Command = std::variant<CommandNoop, CommandExecute, CommandBufferCreate, CommandBufferDelete>;

struct BufferRequirement {
    BufferId buffer_id;
    DeviceId memory_id;
    bool is_write;
};

class Worker: std::enable_shared_from_this<Worker> {
  public:
    friend class TaskContext;

    void make_progress();
    void submit_command(
        JobId id,
        Command command,
        std::vector<JobId> dependencies = {},
        std::vector<BufferRequirement> buffers = {});

  private:
    enum class JobStatus { Pending, Ready, Staging, Scheduled, Done };

    struct Job {
        Job(JobId id, Command kind, std::vector<BufferRequirement> buffers);

        JobId id;
        Command command;
        JobStatus status;
        std::vector<BufferRequirement> buffers;
        size_t requests_pending = 0;
        std::vector<std::shared_ptr<MemoryRequest>> requests = {};
        size_t predecessors_pending = 1;
        std::vector<std::weak_ptr<Job>> predecessors = {};
        std::vector<std::shared_ptr<Job>> successors = {};
    };

    void trigger_predecessor_completed(const std::shared_ptr<Job>&);
    void schedule_task(const std::shared_ptr<Job>&);
    void stage_task(const std::shared_ptr<Job>&);
    void complete_task(const std::shared_ptr<Job>&);

    std::deque<std::shared_ptr<Job>> m_ready_tasks;
    std::unordered_map<JobId, std::shared_ptr<Job>> m_tasks;
    MemoryManager m_memory_manager;
};

struct BufferAccess {
    std::shared_ptr<Allocation> allocation;
};

class TaskContext {
  public:
    TaskContext(
        std::shared_ptr<Worker> worker,
        std::shared_ptr<Worker::Job>,
        std::vector<BufferAccess> buffers);
    ~TaskContext();

  private:
    std::shared_ptr<Worker> m_worker;
    std::shared_ptr<Worker::Job> m_task;
    std::vector<BufferAccess> m_buffers;
};

}  // namespace kmm