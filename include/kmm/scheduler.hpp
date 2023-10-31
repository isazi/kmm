#pragma once

#include <deque>
#include <memory>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "kmm/memory_manager.hpp"
#include "kmm/types.hpp"
#include "memory.hpp"
#include "object_manager.hpp"

namespace kmm {

class ExecutorContext;
class TaskCompletion;
class TaskContext;

struct BufferAccess {
    const Allocation* allocation;
    bool writable = false;
};

class Task {
  public:
    virtual ~Task() = default;
    virtual void execute(ExecutorContext&, TaskContext&) = 0;
};

class Executor {
  public:
    virtual ~Executor() = default;
    virtual void submit(std::shared_ptr<Task>, std::vector<BufferAccess>, TaskCompletion) = 0;
};

struct BufferRequirement {
    PhysicalBufferId buffer_id;
    DeviceId memory_id;
    bool is_write;
};

struct CommandExecute {
    std::optional<ObjectId> output_object_id;
    DeviceId device_id;
    std::shared_ptr<Task> task;
    std::vector<BufferRequirement> buffers;
};

struct CommandNoop {};

struct CommandBufferCreate {
    PhysicalBufferId id;
    BufferLayout description;
};

struct CommandBufferDelete {
    PhysicalBufferId id;
};

struct CommandObjectDelete {
    ObjectId id;
};

using Command = std::variant<
    CommandNoop,
    CommandExecute,
    CommandBufferCreate,
    CommandBufferDelete,
    CommandObjectDelete>;

struct CommandPacket {
    CommandPacket(JobId id, Command command, std::vector<JobId> dependencies) :
        id(id),
        command(std::move(command)),
        dependencies(std::move(dependencies)) {}

    JobId id;
    Command command;
    std::vector<JobId> dependencies;
};

class Scheduler: public std::enable_shared_from_this<Scheduler> {
  public:
    friend class TaskCompletion;

    void make_progress();
    void submit_command(CommandPacket packet);

  private:
    enum class JobStatus { Pending, Ready, Staging, Scheduled, Done };
    struct Job;
    struct JobWaker;

    void trigger_predecessor_completed(const std::shared_ptr<Job>&);
    void schedule_job(const std::shared_ptr<Job>& job);
    void stage_job(const std::shared_ptr<Job>& job);
    void complete_job(const std::shared_ptr<Job>&);

    std::deque<std::shared_ptr<Job>> m_ready_tasks;
    std::unordered_map<JobId, std::shared_ptr<Job>> m_jobs;
    std::vector<std::shared_ptr<Executor>> m_executors;
    std::shared_ptr<MemoryManager> m_memory_manager;
    std::shared_ptr<ObjectManager> m_object_manager;
};

class TaskCompletion {
  public:
    TaskCompletion(std::weak_ptr<Scheduler> worker, std::weak_ptr<Scheduler::Job>);

    TaskCompletion(TaskCompletion&&) noexcept = default;
    TaskCompletion(const TaskContext&) = delete;
    void complete();
    ~TaskCompletion();

  private:
    std::weak_ptr<Scheduler> m_worker;
    std::weak_ptr<Scheduler::Job> m_job;
};

}  // namespace kmm