#pragma once

#include "task_data.hpp"

#include "kmm/core/view.hpp"

namespace kmm {

template<typename T, typename L>
struct PackedArray {
    size_t buffer_index;
    L layout;
};

template<typename T, typename L>
struct TaskDataDeserialize<ExecutionSpace::Host, PackedArray<T, L>> {
    static basic_view<T, L, views::accessors::host> unpack(
        const TaskContext& context,
        PackedArray<T, L> array) {
        T* data = static_cast<T*>(context.accessors.at(array.buffer_index).address);

        return {data, array.layout};
    }
};

template<typename T, typename L>
struct TaskDataDeserialize<ExecutionSpace::Host, PackedArray<const T, L>> {
    static basic_view<const T, L, views::accessors::host> unpack(
        const TaskContext& context,
        PackedArray<const T, L> array) {
        const T* data = static_cast<const T*>(context.accessors.at(array.buffer_index).address);

        return {data, array.layout};
    }
};

template<typename T, typename L>
struct TaskDataDeserialize<ExecutionSpace::Cuda, PackedArray<T, L>> {
    static basic_view<T, L, views::accessors::gpu_device> unpack(
        const TaskContext& context,
        PackedArray<T, L> array) {
        T* data = static_cast<T*>(context.accessors.at(array.buffer_index).address);

        return {data, array.layout};
    }
};

template<typename T, typename L>
struct TaskDataDeserialize<ExecutionSpace::Cuda, PackedArray<const T, L>> {
    static basic_view<const T, L, views::accessors::gpu_device> unpack(
        const TaskContext& context,
        PackedArray<const T, L> array) {
        const T* data = static_cast<const T*>(context.accessors.at(array.buffer_index).address);

        return {data, array.layout};
    }
};

}  // namespace kmm