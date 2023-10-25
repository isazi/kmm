#pragma once

#include <deque>
#include <memory>
#include <unordered_map>
#include <vector>

#include "kmm/memory_manager.hpp"
#include "kmm/types.hpp"

namespace kmm {

class ExecutorContext {};

class TaskContext {};

class Task {
  public:
    virtual ~Task() = default;
    virtual void execute(ExecutorContext&, TaskContext&) = 0;
};

class TaskGraph {
  public:
    void make_progress(MemoryManager& mm);

    void insert_task(
        TaskId id,
        DeviceId device_id,
        std::shared_ptr<Task> task,
        std::vector<BufferRequirement> buffers,
        std::vector<TaskId> dependencies);

  private:
    enum class Status { Pending, Ready, Staging, Scheduled, Done };

    struct Node {
        TaskId id;
        Status status;
        std::shared_ptr<Task> task;
        std::vector<BufferRequirement> buffers;
        size_t predecessors_pending;
        std::vector<std::weak_ptr<Node>> predecessors;
        std::vector<std::shared_ptr<Node>> successors;
    };

    void stage_task(std::shared_ptr<Node>);

    std::unordered_map<TaskId, std::shared_ptr<Node>> m_tasks;
    std::deque<std::shared_ptr<Node>> m_ready_tasks;
};

}  // namespace kmm