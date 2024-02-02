#pragma once

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "kmm/runtime.hpp"

namespace kmm {

enum struct ExecutionSpace { Host, Cuda };

template<ExecutionSpace Space, typename T, typename = void>
struct TaskArgument {
    using type = T;

    static TaskArgument pack(TaskBuilder& reqs, T value) {
        return {std::move(value)};
    }

    T unpack(TaskContext& context) {
        return std::move(value);
    }

    T value;
};

template<ExecutionSpace Space, typename T, typename = void>
struct TaskArgumentPack {
    using type = TaskArgument<Space, T>;

    static type pack(TaskBuilder& reqs, T value) {
        return type::pack(reqs, std::move(value));
    }
};

template<ExecutionSpace Space, typename T, typename = void>
struct TaskArgumentUnpack {};

template<ExecutionSpace Space, typename T>
struct TaskArgumentUnpack<Space, TaskArgument<Space, T>> {
    using type = T;

    static type unpack(TaskContext& context, TaskArgument<Space, T> arg) {
        return arg.unpack(context);
    }
};

template<ExecutionSpace Space, typename T>
using pack_argument_type = typename TaskArgumentPack<Space, std::decay_t<T>>::type;

template<ExecutionSpace Space, typename T>
pack_argument_type<Space, T> pack_argument(Runtime& rt, TaskRequirements& reqs, T&& value) {
    return TaskArgumentPack<Space, std::decay_t<T>>::pack(rt, reqs, std::forward<T>(value));
}

template<ExecutionSpace Space, typename T>
using unpack_argument_type = typename TaskArgumentUnpack<Space, std::decay_t<T>>::type;

template<ExecutionSpace Space, typename T>
unpack_argument_type<Space, std::decay_t<T>> unpack_argument(TaskContext& context, T&& value) {
    return TaskArgumentUnpack<Space, std::decay_t<T>>::unpack(context, value);
}

template<typename Launcher, typename... Args>
class TaskImpl final: public Task {
  public:
    static constexpr ExecutionSpace execution_space = Launcher::execution_space;

    TaskImpl(Launcher launcher, pack_argument_type<execution_space, Args>... args) :
        m_launcher(std::move(launcher)),
        m_args(std::move(args)...) {}

    void execute(Device& device, TaskContext& context) override {
        return execute_impl(device, context, std::index_sequence_for<Args...>());
    }

  private:
    template<size_t... Is>
    void execute_impl(Device& device, TaskContext& context, std::index_sequence<Is...>) {
        m_launcher(
            device,
            context,
            TaskArgumentUnpack<execution_space, pack_argument_type<execution_space, Args>>::unpack(
                context,
                std::move(std::get<Is>(m_args)))...);
    }

    Launcher m_launcher;
    std::tuple<pack_argument_type<execution_space, Args>...> m_args;
};

template<typename Launcher, typename... Args>
EventId submit_task_with_launcher(
    Runtime& rt,
    DeviceId device_id,
    Launcher launcher,
    Args... args) {
    static constexpr ExecutionSpace execution_space = Launcher::execution_space;
    using task_type = TaskImpl<Launcher, std::decay_t<Args>...>;

    auto builder = TaskBuilder(&rt, device_id);

    std::shared_ptr<Task> task = std::make_shared<task_type>(
        launcher,
        TaskArgumentPack<execution_space, Args>::pack(builder, std::move(args))...);

    return builder.submit(std::move(task));
}

template<typename T>
struct Write {
    T* inner;

    T* get() const {
        return inner;
    }

    T& operator*() const {
        return *inner;
    }

    T* operator->() const {
        return inner;
    }
};

template<typename T>
Write(T&) -> Write<T>;

template<typename T>
Write<T> write(T& value) {
    return {&value};
}

}  // namespace kmm