#include "kmm/task_graph.hpp"

#include <stdexcept>
#include <utility>

namespace kmm {

void TaskGraph::insert_task(TaskId id, TaskKind kind, std::vector<TaskId> predecessors) {
    // Remove duplicates from predecessors
    std::sort(predecessors.begin(), predecessors.end());
    auto unique_end = std::unique(predecessors.begin(), predecessors.end());
    predecessors.erase(unique_end, predecessors.end());

    // Allocate task
    auto state = std::make_shared<TaskNode>(TaskNode {
        .id = id,
        .kind = std::move(kind),
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
    auto it = tasks.find(id);
    if (it == tasks.end()) {
        return;  // not found
    }

    auto task = std::move(it->second);
    tasks.erase(it);

    for (const auto& successor : task->successors) {
        successor->predecessors_pending--;

        if (successor->predecessors_pending == 1) {
            ready_tasks.push_back(successor);
        }
    }
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

std::optional<std::shared_ptr<TaskNode>> TaskGraph::pop_ready() {
    if (!ready_tasks.empty()) {
        auto front = std::move(ready_tasks.front());
        ready_tasks.pop_front();
        return front;
    } else {
        return {};
    }
}

}  // namespace kmm