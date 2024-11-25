#pragma once

#include "kmm/api/mapper.hpp"
#include "kmm/api/task_builder.hpp"
#include "kmm/core/task.hpp"

namespace kmm {

enum struct ExecutionSpace { Host, Device };

template<typename T>
struct ArgumentHandler;

template<ExecutionSpace, typename T>
struct ArgumentDeserialize;

template<typename T, typename = void>
struct Argument {
    Argument(T value) : m_value(std::move(value)) {}

    static Argument pack(TaskBuilder& builder, T value) {
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

    ArgumentHandler(T value) : m_value(value) {}

    void initialize(const TaskInit& init) {
        // Nothing to do
    }

    type process_chunk(TaskBuilder& builder) {
        return Argument<T>::pack(builder, m_value);
    }

    void finalize(const TaskResult& result) {
        // Nothing to do
    }

  private:
    T m_value;
};

template<ExecutionSpace Space, typename T>
struct ArgumentDeserialize<Space, Argument<T>> {
    static T unpack(TaskContext& context, Argument<T>& data) {
        return data.template unpack<Space>(context);
    }
};

}  // namespace kmm