#pragma once

#include <functional>
#include <utility>

#include "task.hpp"
#include "types.hpp"

namespace kmm {

struct BufferRequirement {
    BufferId buffer_id;
    MemoryId memory_id;
    AccessMode mode;
};

template<typename Arg>
class TaskArgumentPacker {
  public:
    using type = Arg;

    static type call(Arg input, std::vector<BufferRequirement>& reqs) {
        return input;
    }
};

class ManagerImpl;

class Manager {
  public:
    Manager() = default;
    explicit Manager(std::shared_ptr<ManagerImpl> impl) : impl_(std::move(impl)) {}

    const std::vector<std::shared_ptr<Memory>>& memories() const;
    const std::vector<std::shared_ptr<Executor>>& executors() const;

    BufferId create_buffer(BufferLayout) const;
    void increment_buffer_refcount(BufferId, uint64_t count = 1) const;
    void decrement_buffer_refcount(BufferId, uint64_t count = 1) const;
    void prefetch_buffer(BufferId, MemoryId) const;

    TaskId submit_task(
        ExecutorId executor_id,
        std::shared_ptr<Task> task,
        std::vector<BufferRequirement> reqs,
        std::vector<TaskId> deps) const;
    bool query_task_done(TaskId) const;
    void attach_task_callback(TaskId, std::function<void()> callback) const;

    template<typename D, typename F, typename... Args>
    TaskId submit(D device, F fun, Args... args) const {
        ExecutorId executor_id = device.select_executor(*this);

        std::vector<BufferRequirement> reqs;
        auto task = std::make_shared<
            TaskImpl<D::memory_space, std::decay_t<F>, typename TaskArgumentPacker<Args>::type...>>(
            std::move(device),
            std::move(fun),
            TaskArgumentPacker<Args>::call(std::move(args), reqs)...);

        return submit_task(executor_id, std::move(task), std::move(reqs), {});
    }

  private:
    std::shared_ptr<ManagerImpl> impl_;
};

}  // namespace kmm