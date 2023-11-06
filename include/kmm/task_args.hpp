#pragma once

#include "kmm/executor.hpp"

namespace kmm {

enum struct ExecutionSpace { Host, Cuda };

template<ExecutionSpace Space, typename T, typename = void>
struct TaskArgPack {
    using type = std::decay_t<T>;

    static type call(T value, TaskRequirements& requirements) {
        return value;
    }
};

template<ExecutionSpace Space, typename T, typename = void>
struct TaskArgUnpack {
    static const T& call(const T& value, TaskContext& context) {
        return value;
    }
};

template<ExecutionSpace Space, typename Fun, typename... Args>
class TaskImpl: public Task {
  public:
    TaskImpl(Fun fun, Args... args) : m_fun(std::move(fun)), m_args(std::move(args)...) {}

    TaskResult execute(ExecutorContext& executor, TaskContext& context) override {
        execute_impl(executor, context, std::make_index_sequence<sizeof...(Args)>());

        return {};
    }

  private:
    template<size_t... Is>
    auto execute_impl(ExecutorContext& executor, TaskContext& context, std::index_sequence<Is...>)
        -> decltype(auto) {
        return m_fun(TaskArgUnpack<Space, Args>::call(std::get<Is>(m_args), context)...);
    }

  private:
    Fun m_fun;
    std::tuple<Args...> m_args;
};

template<typename T>
struct Write {
    T inner;
};

template<typename T>
Write<std::decay_t<T>> write(T&& value) {
    return {std::forward<T>(value)};
}

}  // namespace kmm