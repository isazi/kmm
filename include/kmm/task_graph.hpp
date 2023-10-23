#include <deque>
#include <functional>

#include "kmm/manager.hpp"
#include "kmm/task.hpp"
#include "kmm/types.hpp"

namespace kmm {

struct TaskKind {
    ExecutorId executor_id;
    std::shared_ptr<Task> task;
};

struct TaskNode;

struct TaskGraph {
    void insert_task(
        TaskId id,
        TaskKind kind,
        std::vector<BufferRequirement> requirements,
        std::vector<TaskId> predecessors);

    void remove_task(TaskId id);
    bool is_done(TaskId id);
    bool attach_callback(TaskId task_id, std::function<void()>& callback);

  private:
    TaskId next_task_id = 1;
    std::unordered_map<TaskId, std::shared_ptr<TaskNode>> tasks;
    std::deque<std::shared_ptr<TaskNode>> ready_tasks;
};
}  // namespace kmm