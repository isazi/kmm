#include "kmm/manager.hpp"

#include <deque>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace kmm {

struct TaskKind {
    ExecutorId executor_id;
    std::shared_ptr<Task> task;
};

struct TaskNode {
    TaskId id;
    uint64_t predecessors_pending = 0;
    std::vector<std::shared_ptr<TaskNode>> successors = {};
    std::vector<std::function<void()>> callbacks = {};

    TaskKind kind;
    std::vector<BufferRequirement> buffers;
};

struct TaskGraph {
    bool is_done(TaskId id) {
        return tasks.find(id) == tasks.end();
    }

    bool attach_callback(TaskId task_id, std::function<void()>& callback) {
        auto it = tasks.find(task_id);
        if (it == tasks.end()) {
            return false;
        }

        auto& task = it->second;
        task->callbacks.push_back(std::move(callback));
        return true;
    }

    TaskId insert(
        TaskKind kind,
        const std::vector<BufferRequirement>& requirements,
        const std::vector<TaskId>& predecessors) {
        auto id = next_task_id++;
        auto state = std::make_shared<TaskNode>(TaskNode {
            .id = id,
            .kind = std::move(kind),
            .buffers = requirements,
        });

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

        return id;
    }

    void remove(TaskId id) {
        throw std::runtime_error("todo");
    }

    TaskId next_task_id = 1;
    std::unordered_map<TaskId, std::shared_ptr<TaskNode>> tasks;
    std::deque<std::shared_ptr<TaskNode>> ready_tasks;
    std::vector<std::shared_ptr<Executor>> executors;
};

struct BufferState {
    BufferState(BufferId id, BufferLayout layout) : id_(id), ref_count_(1), layout_(layout) {}

    void increment_ref_counter(uint64_t count) {
        if (ref_count_ == 0 || ref_count_ >= std::numeric_limits<uint64_t>::max() - count) {
            throw std::runtime_error("invalid buffer ref count");
        }

        ref_count_ += count;
    }

    bool decrement_ref_counter(uint64_t count) {
        if (ref_count_ < count) {
            throw std::runtime_error("invalid buffer ref count");
        }

        ref_count_ -= count;
        return ref_count_ > 0;
    }

    void find_access_dependencies(AccessMode mode, std::vector<TaskId>& deps_out) const {
        switch (mode) {
            case AccessMode::Read:
                deps_out.insert(deps_out.end(), last_writers.begin(), last_writers.end());
                break;
            case AccessMode::Write:
                deps_out.insert(deps_out.end(), last_writers.begin(), last_writers.end());
                deps_out.insert(deps_out.end(), last_readers.begin(), last_readers.end());
                break;
        }
    }

    void update_access_dependencies(AccessMode mode, TaskId task_id) {
        switch (mode) {
            case AccessMode::Read:
                last_readers.push_back(task_id);
                break;
            case AccessMode::Write:
                last_readers = {};
                last_writers = {task_id};
                break;
        }
    }

    BufferId id_;
    uint64_t ref_count_;
    BufferLayout layout_;
    std::vector<TaskId> last_readers;
    std::vector<TaskId> last_writers;
};

struct BufferManager {
    BufferId create(BufferLayout layout) {
        BufferId id = next_buffer_id;
        auto state = std::make_shared<BufferState>(id, layout);
        buffers.emplace(id, state);

        return id;
    }

    std::shared_ptr<BufferState>& get(BufferId id) {
        return buffers.at(id);
    }

    void increment_refcount(BufferId id, uint64_t count) {
        buffers.at(id)->increment_ref_counter(count);
    }

    void decrement_buffer_refcount(BufferId id, uint64_t count) {
        auto& state = buffers.at(id);

        if (state->decrement_ref_counter(count)) {
            return;
        }

        throw std::runtime_error("todo: remove buffer");
    }

    BufferId next_buffer_id = 1;
    std::vector<std::shared_ptr<Memory>> memories;
    std::unordered_map<BufferId, std::shared_ptr<BufferState>> buffers;
};

class ManagerImpl: std::enable_shared_from_this<ManagerImpl> {
  public:
    std::mutex mutex;
    TaskGraph task_graph;
    BufferManager buffer_manager;
};

const std::vector<std::shared_ptr<Memory>>& Manager::memories() const {
    return impl_->buffer_manager.memories;
}

const std::vector<std::shared_ptr<Executor>>& Manager::executors() const {
    return impl_->task_graph.executors;
}

BufferId Manager::create_buffer(BufferLayout layout) const {
    std::lock_guard guard {impl_->mutex};
    return impl_->buffer_manager.create(layout);
}

void Manager::increment_buffer_refcount(BufferId buffer_id, uint64_t count) const {
    std::lock_guard guard {impl_->mutex};
    return impl_->buffer_manager.increment_refcount(buffer_id, count);
}

void Manager::decrement_buffer_refcount(BufferId buffer_id, uint64_t count) const {
    std::lock_guard guard {impl_->mutex};
    return impl_->buffer_manager.decrement_buffer_refcount(buffer_id, count);
}

void Manager::prefetch_buffer(BufferId buffer_id, MemoryId memory_id) const {
    std::lock_guard guard {impl_->mutex};
    throw std::runtime_error("todo");
}

TaskId Manager::submit_task(
    ExecutorId executor_id,
    std::shared_ptr<Task> task,
    std::vector<BufferRequirement> requirements,
    std::vector<TaskId> dependencies) const {
    std::lock_guard guard {impl_->mutex};

    for (const auto& req : requirements) {
        auto& buffer = impl_->buffer_manager.get(req.buffer_id);
        buffer->find_access_dependencies(req.mode, dependencies);
    }

    TaskKind kind = {
        .executor_id = executor_id,
        .task = std::move(task),
    };

    TaskId task_id = impl_->task_graph.insert(std::move(kind), requirements, dependencies);

    for (const auto& req : requirements) {
        auto& buffer = impl_->buffer_manager.get(req.buffer_id);
        buffer->update_access_dependencies(req.mode, task_id);
    }

    //

    return task_id;
}

bool Manager::query_task_done(TaskId task_id) const {
    std::lock_guard guard {impl_->mutex};
    return impl_->task_graph.is_done(task_id);
}

void Manager::attach_task_callback(TaskId task_id, std::function<void()> callback) const {
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