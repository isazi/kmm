#pragma once
#include "access.hpp"
#include "runtime_impl.hpp"

#include "kmm/api/task_data.hpp"
#include "kmm/core/identifiers.hpp"
#include "kmm/core/view.hpp"

namespace kmm {

template<typename T, size_t N>
struct PackedArray {
    size_t buffer_index;
    point<N> offset;
    dim<N> sizes;
};

template<typename T, size_t N>
struct TaskDataDeserializer<ExecutionSpace::Host, PackedArray<T, N>> {
    view_mut<T, N> deserialize(TaskContext& context, PackedArray<T, N> arg) {
        T* addr = static_cast<T*>(context.accessors.at(arg.buffer_index).address);
        return {addr};
    }
};

template<typename T, size_t N>
struct TaskDataDeserializer<ExecutionSpace::Host, PackedArray<const T, N>> {
    view<T, N> deserialize(TaskContext& context, PackedArray<const T, N> arg) {
        const T* addr = static_cast<const T*>(context.accessors.at(arg.buffer_index).address);
        return {addr};
    }
};

template<typename T, size_t N>
struct TaskDataDeserializer<ExecutionSpace::Cuda, PackedArray<T, N>> {
    cuda_view_mut<T, N> deserialize(TaskContext& context, PackedArray<T, N> arg) {
        T* addr = static_cast<T*>(context.accessors.at(arg.buffer_index).address);
        return {addr};
    }
};

template<typename T, size_t N>
struct TaskDataDeserializer<ExecutionSpace::Cuda, PackedArray<const T, N>> {
    cuda_view<T, N> deserialize(TaskContext& context, PackedArray<const T, N> arg) {
        const T* addr = static_cast<const T*>(context.accessors.at(arg.buffer_index).address);
        return {addr};
    }
};

}  // namespace kmm