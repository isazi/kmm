#include "kmm/task_graph.hpp"

#include <functional>
#include <utility>

#include "kmm/memory_manager.hpp"

namespace kmm {
void TaskGraph::make_progress(MemoryManager& mm) {
    while (!m_ready_tasks.empty()) {
        auto task = std::move(m_ready_tasks.front());
        m_ready_tasks.pop_front();

        for (const auto& buffer : task->buffers) {
        }
        //TODO
    }
}

void TaskGraph::insert_task(
    kmm::TaskId id,
    kmm::DeviceId device_id,
    std::shared_ptr<Task> task,
    std::vector<BufferRequirement> buffers,
    std::vector<TaskId> dependencies) {
    // Remove duplicates
    std::sort(dependencies.begin(), dependencies.end());
    auto last = std::unique(dependencies.begin(), dependencies.end());
    dependencies.erase(last, dependencies.end());

    auto predecessors = std::vector<std::weak_ptr<Node>> {};
    predecessors.reserve(dependencies.size());

    auto node = std::make_shared<Node>(Node {
        .id = id,
        .status = Status::Pending,
        .task = std::move(task),
        .buffers = std::move(buffers),
        .predecessors_pending = 0,
        .predecessors = {},
        .successors = {}});

    for (auto dep_id : dependencies) {
        auto it = m_tasks.find(dep_id);
        if (it == m_tasks.end()) {
            continue;
        }

        predecessors.push_back(it->second);
    }

    size_t predecessors_pending = predecessors.size();
    node->predecessors = std::move(predecessors);
    node->predecessors_pending = predecessors_pending;

    if (predecessors_pending == 0) {
        node->status = Status::Ready;
        m_ready_tasks.push_back(std::move(node));
    }
}
}  // namespace kmm
