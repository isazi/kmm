#include "kmm/worker.hpp"

#include <functional>
#include <utility>

#include "kmm/memory_manager.hpp"

namespace kmm {
void Worker::make_progress() {
    while (true) {
        if (auto req = m_memory_manager.poll(); req) {
            auto task = std::dynamic_pointer_cast<Node>(*req);
            task->requests_pending -= 1;

            if (task->requests_pending == 0) {
                schedule_task(std::move(task));
            }
            continue;
        }

        if (!m_ready_tasks.empty()) {
            auto task = std::move(m_ready_tasks.front());
            m_ready_tasks.pop_front();

            stage_task(task);
            continue;
        }

        break;
    }
}

void Worker::trigger_predecessor_completed(std::shared_ptr<Worker::Node> task) {
    task->predecessors_pending -= 1;

    if (task->predecessors_pending == 0) {
        task->status = Status::Ready;
        m_ready_tasks.push_back(std::move(task));
    }
}

void Worker::stage_task(std::shared_ptr<Worker::Node> task) {
    auto requests = std::vector<std::shared_ptr<MemoryRequest>> {};

    for (const auto& arg : task->buffers) {
        requests.push_back(
            m_memory_manager.acquire_buffer(arg.buffer_id, task->device_id, arg.is_write, task));
    }

    size_t requests_pending = requests.size();
    task->status = Status::Staging;
    task->requests_pending = requests_pending;
    task->requests = std::move(requests);

    if (requests_pending == 0) {
        schedule_task(std::move(task));
    }
}

void Worker::schedule_task(std::shared_ptr<Node> task) {
    task->status = Status::Scheduled;

    std::vector<std::shared_ptr<Allocation>> allocations;
    for (const auto& req : task->requests) {
        allocations.push_back(m_memory_manager.view_buffer(req));
    }

    // TODO

    this->complete_task(task);
}

void Worker::complete_task(std::shared_ptr<Node> task) {
    task->status = Status::Done;
    m_tasks.erase(task->id);

    for (const auto& request : task->requests) {
        m_memory_manager.release_buffer(request, std::nullopt);
    }

    for (const auto& successor : task->successors) {
        trigger_predecessor_completed(successor);
    }
}

void Worker::insert_task(
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

    auto new_task = std::make_shared<Node>(Node(id, device_id, task, buffers));

    for (auto dep_id : dependencies) {
        auto it = m_tasks.find(dep_id);
        if (it == m_tasks.end()) {
            continue;
        }

        auto& predecessor = it->second;
        predecessor->successors.push_back(new_task);
        predecessors.push_back(predecessor);
    }

    size_t predecessors_pending = predecessors.size();
    new_task->predecessors = std::move(predecessors);
    new_task->predecessors_pending = predecessors_pending + 1;

    // We always add one "phantom" predecessor to `predecessors_pending` so we can trigger it here
    trigger_predecessor_completed(std::move(new_task));
}

void Worker::create_buffer(BufferId id, const BufferDescription& description) {
    m_memory_manager.create_buffer(id, description);
}

void Worker::delete_buffer(BufferId buffer_id, std::vector<TaskId> dependencies) {}

void Worker::prefetch_buffer(
    BufferId buffer_id,
    DeviceId device_id,
    std::vector<TaskId> dependencies) {}

Worker::Node::Node(
    TaskId id,
    DeviceId device_id,
    std::shared_ptr<Task> task,
    std::vector<BufferRequirement> buffers) :
    id(id),
    status(Worker::Status::Pending),
    device_id(device_id),
    task(std::move(task)),
    buffers(std::move(buffers)) {}

TaskContext::TaskContext(
    std::shared_ptr<Worker> worker,
    std::shared_ptr<Worker::Node> task,
    std::vector<BufferAccess> buffers) :
    m_worker(std::move(worker)),
    m_task(std::move(task)),
    m_buffers(std::move(buffers)) {}

TaskContext::~TaskContext() {
    if (m_task) {
        m_worker->complete_task(m_task);
    }
}

}  // namespace kmm
