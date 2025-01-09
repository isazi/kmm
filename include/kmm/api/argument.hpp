#pragma once

#include "kmm/api/mapper.hpp"
#include "kmm/api/task_builder.hpp"
#include "kmm/core/task.hpp"

namespace kmm {

template<typename T>
struct ArgumentHandler;

template<ExecutionSpace, typename T>
struct ArgumentUnpack;

template<typename T, typename = void>
struct Argument {
    Argument(T value) : m_value(std::move(value)) {}

    static Argument pack(TaskInstance& builder, T value) {
        return {std::move(value)};
    }

    template<ExecutionSpace Space>
    T unpack(TaskContext& context) {
        return m_value;
    }

  private:
    T m_value;
};

template<typename T>
struct ArgumentHandler {
    using type = Argument<T>;

    ArgumentHandler(T value) : m_value(std::move(value)) {}

    void initialize(const TaskGroupInfo& init) {
        // Nothing to do
    }

    type process_chunk(TaskInstance& builder) {
        return Argument<T>::pack(builder, m_value);
    }

    void finalize(const TaskGroupResult& result) {
        // Nothing to do
    }

  private:
    T m_value;
};

template<ExecutionSpace Space, typename T>
struct ArgumentUnpack<Space, Argument<T>> {
    static auto unpack(TaskContext& context, Argument<T>& data) {
        return data.template unpack<Space>(context);
    }
};

template<typename T>
using packed_argument_t = typename ArgumentHandler<std::decay_t<T>>::type;

template<typename T>
packed_argument_t<T> pack_argument(TaskInstance& task, T&& arg) {
    return ArgumentHandler<std::decay_t<T>>(std::forward<T>(arg)).process_chunk(task);
}

template<ExecutionSpace execution_space, typename T>
auto unpack_argument(TaskContext& builder, T&& arg) {
    return ArgumentUnpack<execution_space, std::decay_t<T>>::unpack(builder, std::forward<T>(arg));
}

}  // namespace kmm