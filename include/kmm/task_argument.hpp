#pragma once

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "kmm/runtime_impl.hpp"

namespace kmm {

enum struct ExecutionSpace { Host, Cuda };

template<ExecutionSpace Space, typename T, typename = void>
struct TaskArgument {
    using type = T;

    void pack(RuntimeImpl& rt, TaskRequirements& reqs, T value) {
        m_value = std::move(value);
    }

    T unpack(TaskContext& context) {
        return std::move(m_value);
    }

  private:
    T m_value;
};

template<ExecutionSpace Space, typename T, typename = void>
struct TaskArgumentDelegate {
    using packed_type = TaskArgument<Space, T>;
    using unpacked_type = typename packed_type::type;

    static packed_type pack(RuntimeImpl& rt, TaskRequirements& reqs, T value) {
        TaskArgument<Space, T> arg;
        arg.pack(rt, reqs, std::move(value));
        return arg;
    }

    static void post_submission(RuntimeImpl& rt, EventId id, const T& value, packed_type arg) {}

    static unpacked_type unpack(TaskContext& context, packed_type arg) {
        return arg.unpack(context);
    }
};

template<ExecutionSpace Space, typename T>
using packed_task_argument_type =
    typename TaskArgumentDelegate<Space, std::decay_t<T>>::packed_type;

template<ExecutionSpace Space, typename T>
using task_argument_type = typename TaskArgumentDelegate<Space, std::decay_t<T>>::unpacked_type;

template<typename Launcher, typename... Args>
class TaskImpl final: public Task {
  public:
    static constexpr ExecutionSpace execution_space = Launcher::execution_space;

    TaskImpl(
        Launcher launcher,
        std::tuple<packed_task_argument_type<execution_space, Args>...> args) :
        m_launcher(std::move(launcher)),
        m_args(std::move(args)) {}

    void execute(Device& device, TaskContext& context) override {
        return execute_impl(device, context, std::index_sequence_for<Args...>());
    }

  private:
    template<size_t... Is>
    void execute_impl(Device& device, TaskContext& context, std::index_sequence<Is...>) {
        m_launcher(
            device,
            context,
            TaskArgumentDelegate<execution_space, Args>::unpack(
                context,
                std::move(std::get<Is>(m_args)))...);
    }

    Launcher m_launcher;
    std::tuple<packed_task_argument_type<execution_space, Args>...> m_args;
};

template<typename Launcher, typename... Args>
struct TaskLaunchHelper {
    static constexpr ExecutionSpace execution_space = Launcher::execution_space;

    template<size_t... Is>
    static EventId call(
        std::index_sequence<Is...>,
        Launcher launcher,
        DeviceId device_id,
        RuntimeImpl& rt,
        Args... args) {
        auto reqs = TaskRequirements(device_id);
        auto packed_args = std::tuple<packed_task_argument_type<execution_space, Args>...> {
            TaskArgumentDelegate<execution_space, Args>::pack(rt, reqs, args)...};

        std::shared_ptr<Task> task =
            std::make_shared<TaskImpl<Launcher, std::decay_t<Args>...>>(launcher, packed_args);

        auto event_id = rt.submit_task(std::move(task), std::move(reqs));
        (TaskArgumentDelegate<execution_space, Args>::post_submission(
             rt,
             event_id,
             args,
             std::get<Is>(packed_args)),
         ...);

        return event_id;
    }
};

template<typename Launcher, typename... Args>
EventId submit_task_with_launcher(
    RuntimeImpl& rt,
    DeviceId device_id,
    Launcher launcher,
    Args... args) {
    return TaskLaunchHelper<Launcher, Args...>::call(
        std::index_sequence_for<Args...>(),
        launcher,
        device_id,
        rt,
        args...);
}

template<typename T>
struct Write {
    T& inner;
};

template<typename T>
Write<T> write(T& value) {
    return {value};
}

}  // namespace kmm