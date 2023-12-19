#pragma once

#include "kmm/executor.hpp"

namespace kmm {

enum struct ExecutionSpace { Host, Cuda };

template<ExecutionSpace Space, typename T, typename = void>
struct TaskArgumentSerializer {
    using type = std::decay_t<T>;

    type serialize(T value, TaskRequirements& requirements) {
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

    TaskResult execute(ExecutorContext& executor, TaskContext& context) override {
        try {
            execute_impl(executor, context, std::make_index_sequence<sizeof...(Args)>());
            return {};
        } catch (const std::exception& e) {
            return TaskError(e);
        }
    }

  private:
    template<size_t... Is>
    void execute_impl(ExecutorContext& executor, TaskContext& context, std::index_sequence<Is...>) {
        m_fun(
            TaskArgumentDeserializer<Space, Args>().deserialize(std::get<Is>(m_args), context)...);
    }

    Fun m_fun;
    std::tuple<Args...> m_args;
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