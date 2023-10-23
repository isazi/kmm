#include "kmm/runtime.hpp"

#include <deque>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "kmm/buffer_manager.hpp"
#include "kmm/memory_manager.hpp"
#include "kmm/task_graph.hpp"

namespace kmm {

class RuntimeImpl: std::enable_shared_from_this<RuntimeImpl> {
  public:
    std::mutex mutex;
    TaskGraph task_graph;
    BufferManager buffer_manager;
    TaskId next_task_id = 1;
    MemoryManager memory_manager;
    std::vector<std::shared_ptr<Executor>> executors;
};

const std::vector<std::shared_ptr<Memory>>& Runtime::memories() const {
    return impl_->memory_manager.memories;
}

const std::vector<std::shared_ptr<Executor>>& Runtime::executors() const {
    return impl_->executors;
}

BufferId Runtime::create_buffer(BufferLayout layout, MemoryId home) const {
    std::lock_guard guard {impl_->mutex};
    BufferId id = impl_->buffer_manager.create(layout, home);
    return id;
}

void Runtime::increment_buffer_refcount(BufferId buffer_id, uint64_t count) const {
    std::lock_guard guard {impl_->mutex};
    return impl_->buffer_manager.increment_refcount(buffer_id, count);
}

void Runtime::decrement_buffer_refcount(BufferId buffer_id, uint64_t count) const {
    std::lock_guard guard {impl_->mutex};

    if (!impl_->buffer_manager.decrement_refcount(buffer_id, count)) {
        return;
    }

    throw std::runtime_error("TODO");
}

void Runtime::prefetch_buffer(BufferId buffer_id, MemoryId memory_id) const {
    std::lock_guard guard {impl_->mutex};
    throw std::runtime_error("todo");
}

TaskId Runtime::submit_task(
    ExecutorId executor_id,
    std::shared_ptr<Task> task,
    std::vector<BufferRequirement> requirements,
    std::vector<TaskId> dependencies) const {
    std::lock_guard guard {impl_->mutex};

    TaskId task_id = impl_->next_task_id++;

    for (const auto& req : requirements) {
        impl_->buffer_manager.update_access(req.buffer_id, task_id, req.mode, dependencies);
    }

    TaskKind kind = {
        .executor_id = executor_id,
        .task = std::move(task),
        .buffers = std::move(requirements)};

    impl_->task_graph.insert_task(task_id, std::move(kind), std::move(dependencies));
    return task_id;
}

bool Runtime::query_task_done(TaskId task_id) const {
    std::lock_guard guard {impl_->mutex};
    return impl_->task_graph.is_done(task_id);
}

void Runtime::attach_task_callback(TaskId task_id, std::function<void()> callback) const {
    bool task_completed;

    {
        std::lock_guard guard {impl_->mutex};
        task_completed = !impl_->task_graph.attach_callback(task_id, callback);
    }

    // We need to invoke the callback without holding the lock.
    if (task_completed) {
        callback();
    }
}

}  // namespace kmm