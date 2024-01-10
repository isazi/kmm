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

template<typename Launcher, typename... Args>
class TaskImpl final: public Task {
  public:
    static constexpr ExecutionSpace execution_space = Launcher::execution_space;

    TaskImpl(Launcher launcher, Args... args) :
        m_launcher(std::move(launcher)),
        m_args(std::move(args)...) {}

    void execute(Executor& executor, TaskContext& context) override {
        return execute_impl(executor, context, std::index_sequence_for<Args...>());
    }

  private:
    template<size_t... Is>
    void execute_impl(Executor& executor, TaskContext& context, std::index_sequence<Is...>) {
        m_launcher(
            executor,
            context,
            TaskArgumentDeserializer<execution_space, Args>().deserialize(  //
                std::get<Is>(m_args),
                context)...);
    }

    Launcher m_launcher;
    std::tuple<Args...> m_args;
};

template<typename Launcher, typename... Args>
struct TaskLauncher {
    static constexpr ExecutionSpace execution_space = Launcher::execution_space;

    static EventId call(Launcher launcher, ExecutorId executor_id, RuntimeImpl& rt, Args... args) {
        return call_impl(std::index_sequence_for<Args...>(), launcher, executor_id, rt, args...);
    }

  private:
    template<size_t... Is>
    static EventId call_impl(
        std::index_sequence<Is...>,
        Launcher launcher,
        ExecutorId executor_id,
        RuntimeImpl& rt,
        Args... args) {
        auto reqs = TaskRequirements(executor_id);
        auto serializers =
            std::tuple<TaskArgumentSerializer<execution_space, std::decay_t<Args>>...>();

        std::shared_ptr<Task> task = std::make_shared<TaskImpl<
            Launcher,
            typename TaskArgumentSerializer<execution_space, std::decay_t<Args>>::type...>>(
            launcher,
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