#pragma once

#include <tuple>

#include "access.hpp"
#include "runtime_impl.hpp"

#include "kmm/core/geometry.hpp"

namespace kmm {

enum struct ExecutionSpace { Host, Cuda };

template<typename T, typename = void>
struct TaskData {
    TaskData(T value) : m_value(std::move(value)) {}

    static TaskData pack(TaskRequirements& req, T value) {
        return {std::move(value)};
    }

    template<ExecutionSpace Space>
    T unpack(TaskContext& context) {
        return m_value;
    }

  private:
    T m_value;
};

template<typename T, typename = void>
struct TaskDataProcessor {
    using type = TaskData<T>;

    TaskDataProcessor(T value) : m_value(value) {}

    template<size_t N>
    type pre_enqueue(Chunk<N> chunk, TaskRequirements& req) {
        return TaskData<T>::pack(req, m_value);
    }

    template<size_t N>
    void post_enqueue(Chunk<N> chunk, TaskResult& result) {}

    void finalize() {}

  private:
    T m_value;
};

template<ExecutionSpace Space, typename T, typename = void>
struct TaskDataDeserializer {};

template<ExecutionSpace Space, typename T>
struct TaskDataDeserializer<Space, TaskData<T>> {
    auto deserialize(TaskContext& context, TaskData<T>& arg) {
        return arg.template unpack<Space>(context);
    }
};

template<typename T>
using serialized_argument_t = typename TaskDataProcessor<T>::type;

template<typename T>
serialized_argument_t<T> serialize_argument(TaskRequirements& req, T&& arg) {
    auto domain = rect<0>();
    auto proc = TaskDataProcessor<T> {std::forward<T>(arg)};
    return proc.pre_enqueue(domain, req);
}

template<typename T, typename P = FullMapping>
struct Read {
    T inner;
    P index_mapping;
};

template<typename T, typename P = FullMapping>
Read<T, P> read(T& that, P mapping = {}) {
    return {that, mapping};
}

template<typename T, typename P = FullMapping>
struct Write {
    T& inner;
    P index_mapping;
};

template<typename T, typename P = FullMapping>
Write<T, P> write(T& that, P mapping = {}) {
    return {that, mapping};
}

}  // namespace kmm