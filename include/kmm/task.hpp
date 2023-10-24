#pragma once

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "kmm/types.hpp"

namespace kmm {

class ExecutorContext {
  public:
    virtual ~ExecutorContext() = default;
};

struct BufferAccess {
    BufferId buffer_id = INVALID_BUFFER_ID;
    MemoryId memory_id = INVALID_MEMORY_ID;
    bool is_writable = false;
    std::shared_ptr<Allocation> buffer;
};

class TaskContext {
    std::vector<BufferAccess> buffers;
};

template<MemorySpace Space, typename Arg>
class TaskArgumentUnpacker {
  public:
    using type = Arg;

    static constexpr type call(const TaskContext& ctx, Arg arg) {
        return arg;
    }
};

class Task {
  public:
    virtual ~Task() = default;
    virtual void execute(const ExecutorContext& executor, TaskContext ctx) const = 0;
};

template<MemorySpace Space, typename Fun, typename... Args>
class TaskImpl: public Task {
  public:
    void execute(const ExecutorContext& executor, TaskContext ctx) const override {
        execute_with_sequence(executor, std::move(ctx), std::index_sequence_for<Args...>());
    }

  private:
    template<size_t... Is>
    void execute_with_sequence(
        const ExecutorContext& executor,
        TaskContext ctx,
        std::index_sequence<Is...>) const {
        function_(TaskArgumentUnpacker<Space, std::tuple_element<Is, std::tuple<Args...>>>::call(
            ctx,
            std::get<Is>(args_))...);
    }

    Fun function_;
    std::tuple<Args...> args_;
};

class Executor {
  public:
    virtual ~Executor() = default;
    virtual std::string name() const = 0;
    virtual std::optional<MemoryId> memory_affinity() const {
        return {};
    }

    virtual void enqueue_task(std::shared_ptr<Task> task, TaskContext ctx) const = 0;
};

}  // namespace kmm