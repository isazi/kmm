#pragma once

#include <functional>
#include <utility>

#include "task.hpp"
#include "types.hpp"

namespace kmm {

template<typename Arg>
class TaskArgumentPacker {
  public:
    using type = Arg;

    static type call(Arg input, std::vector<BufferRequirement>& reqs) {
        return input;
    }
};

class RuntimeImpl;

class Runtime {
  public:
    Runtime(const Runtime&) = default;
    Runtime(std::shared_ptr<RuntimeImpl> impl) : impl_(std::move(impl)) {}

    bool operator==(const Runtime& that) const {
        return this->impl_ == that.impl_;
    }

    bool operator!=(const Runtime& that) const {
        return !(*this == that);
    }

    const std::vector<std::shared_ptr<Memory>>& memories() const;
    const std::vector<std::shared_ptr<Executor>>& executors() const;

    BufferId create_buffer(BufferLayout, MemoryId home = 0) const;
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
    std::shared_ptr<RuntimeImpl> impl_;
};

}  // namespace kmm