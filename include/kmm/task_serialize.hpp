#pragma once

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "kmm/runtime_impl.hpp"

namespace kmm {

enum struct ExecutionSpace { Host, Cuda };

template<ExecutionSpace Space, typename T, typename = void>
struct TaskArgumentSerializer {
    using type = std::decay_t<T>;

    type serialize(RuntimeImpl& rt, T value, TaskRequirements& requirements) {
        return std::forward<T>(value);
    }

    void update(RuntimeImpl& rt, EventId id) {}
};

template<ExecutionSpace Space, typename T, typename = void>
struct TaskArgumentDeserializer {
    const T& deserialize(const T& value, TaskContext& context) {
        return value;
    }
};

template<ExecutionSpace Space, typename Fun, typename... Args>
class TaskImpl: public Task {
  public:
    TaskImpl(Fun fun, Args... args) : m_fun(std::move(fun)), m_args(std::move(args)...) {}

    void execute(ExecutorContext& executor, TaskContext& context) override {
        return execute_impl(executor, context, std::index_sequence_for<Args...>());
    }

  private:
    template<size_t... Is>
    void execute_impl(ExecutorContext& executor, TaskContext& context, std::index_sequence<Is...>) {
        m_fun(TaskArgumentDeserializer<Space, Args>().deserialize(  //
            std::get<Is>(m_args),
            context)...);
    }

    Fun m_fun;
    std::tuple<Args...> m_args;
};

template<ExecutionSpace Space, typename Fun, typename... Args>
struct TaskLauncher {
    static EventId call(ExecutorId executor_id, RuntimeImpl& rt, Fun fun, Args... args) {
        return call_impl(std::index_sequence_for<Args...>(), executor_id, rt, fun, args...);
    }

  private:
    template<size_t... Is>
    static EventId call_impl(
        std::index_sequence<Is...>,
        ExecutorId executor_id,
        RuntimeImpl& rt,
        Fun fun,
        Args... args) {
        auto reqs = TaskRequirements(executor_id);
        auto serializers = std::tuple<TaskArgumentSerializer<Space, std::decay_t<Args>>...>();

        std::shared_ptr<Task> task = std::make_shared<TaskImpl<
            Space,
            std::decay_t<Fun>,
            typename TaskArgumentSerializer<Space, std::decay_t<Args>>::type...>>(
            std::forward<Fun>(fun),
            std::get<Is>(serializers).serialize(rt, std::forward<Args>(args), reqs)...);

        auto event_id = rt.submit_task(std::move(task), std::move(reqs));
        (std::get<Is>(serializers).update(rt, event_id), ...);

        return event_id;
    }
};

template<typename T>
struct Write {
    T& inner;
};

template<typename T>
Write<T> write(T& value) {
    return {value};
}

}  // namespace kmm