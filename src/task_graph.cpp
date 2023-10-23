#include "kmm/task_graph.hpp"

#include <stdexcept>
#include <utility>

namespace kmm {

struct TaskNode {
    TaskId id;
    uint64_t predecessors_pending = 0;
    std::vector<std::shared_ptr<TaskNode>> successors = {};
    std::vector<std::function<void()>> callbacks = {};
    TaskKind kind;
    std::vector<BufferRequirement> buffers;
};

void TaskGraph::insert_task(
    TaskId id,
    TaskKind kind,
    std::vector<BufferRequirement> requirements,
    std::vector<TaskId> predecessors) {
    // Remove duplicates from predecessors
    std::sort(predecessors.begin(), predecessors.end());
    auto unique_end = std::unique(predecessors.begin(), predecessors.end());
    predecessors.erase(unique_end, predecessors.end());

    // Allocate task
    auto state = std::make_shared<TaskNode>(TaskNode {
        .id = id,
        .kind = std::move(kind),
        .buffers = std::move(requirements),
    });

    tasks.insert({id, state});

    // iterate over predecessors
    uint64_t predecessors_pending = 0;

    for (auto predecessor_id : predecessors) {
        auto it = tasks.find(predecessor_id);
        if (it == tasks.end()) {
            continue;
        }

        auto& predecessor = it->second;
        predecessor->successors.push_back(state);
        predecessors_pending++;
    }

    state->predecessors_pending = predecessors_pending;

    if (predecessors_pending == 0) {
        ready_tasks.push_back(std::move(state));
    }
}

void TaskGraph::remove_task(TaskId id) {
    throw std::runtime_error("todo");
}

bool TaskGraph::is_done(TaskId id) {
    return tasks.find(id) == tasks.end();
}

bool TaskGraph::attach_callback(TaskId task_id, std::function<void()>& callback) {
    auto it = tasks.find(task_id);
    if (it == tasks.end()) {
        return false;
    }

    auto& task = it->second;
    task->callbacks.push_back(std::move(callback));
    return true;
}

}  // namespace kmm