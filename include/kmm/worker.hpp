#pragma once

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include "kmm/memory_manager.hpp"
#include "kmm/types.hpp"

namespace kmm {

class Worker;

struct BufferAccess {
    std::shared_ptr<Allocation> allocation;
};

class ExecutorContext {};
class TaskContext;

class Task {
  public:
    virtual ~Task() = default;
    virtual void execute(ExecutorContext&, TaskContext&) = 0;
};

struct BufferRequirement {
    BufferId buffer_id;
    bool is_write;
};

class Worker: std::enable_shared_from_this<Worker> {
  public:
    friend class TaskContext;

    void make_progress();

    void insert_task(
        TaskId id,
        DeviceId device_id,
        std::shared_ptr<Task> task,
        std::vector<BufferRequirement> buffers,
        std::vector<TaskId> dependencies);

    void create_buffer(BufferId buffer_id, const BufferDescription& description);

    void delete_buffer(BufferId buffer_id, std::vector<TaskId> dependencies);

    void prefetch_buffer(BufferId buffer_id, DeviceId device_id, std::vector<TaskId> dependencies);

  private:
    enum class Status { Pending, Ready, Staging, Scheduled, Done };

    struct Node {
        Node(
            TaskId id,
            DeviceId device_id,
            std::shared_ptr<Task> task,
            std::vector<BufferRequirement> buffers);

        TaskId id;
        Status status;
        DeviceId device_id;
        std::shared_ptr<Task> task;
        std::vector<BufferRequirement> buffers;
        size_t requests_pending = 0;
        std::vector<std::shared_ptr<MemoryRequest>> requests = {};
        size_t predecessors_pending = 1;
        std::vector<std::weak_ptr<Node>> predecessors = {};
        std::vector<std::shared_ptr<Node>> successors = {};
    };

    void trigger_predecessor_completed(std::shared_ptr<Node> task);
    void schedule_task(std::shared_ptr<Node>);
    void stage_task(std::shared_ptr<Node>);
    void complete_task(std::shared_ptr<Node>);

    std::deque<std::shared_ptr<Node>> m_ready_tasks;
    std::unordered_map<TaskId, std::shared_ptr<Node>> m_tasks;
    MemoryManager m_memory_manager;
};

class TaskContext {
  public:
    TaskContext(
        std::shared_ptr<Worker> worker,
        std::shared_ptr<Worker::Node>,
        std::vector<BufferAccess> buffers);
    ~TaskContext();

  private:
    std::shared_ptr<Worker> m_worker;
    std::shared_ptr<Worker::Node> m_task;
    std::vector<BufferAccess> m_buffers;
};

}  // namespace kmm