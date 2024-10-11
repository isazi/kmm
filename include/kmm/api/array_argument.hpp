#pragma once

#include "argument.hpp"

#include "kmm/core/view.hpp"

namespace kmm {

template<typename T, typename L>
struct ArrayArgument {
    size_t buffer_index;
    L layout;
};

template<typename T, typename L>
struct ArgumentDeserialize<ExecutionSpace::Host, ArrayArgument<T, L>> {
    static basic_view<T, L, views::accessors::host> unpack(
        const TaskContext& context,
        ArrayArgument<T, L> array
    ) {
        T* data = static_cast<T*>(context.accessors.at(array.buffer_index).address);

        return {data, array.layout};
    }
};

template<typename T, typename L>
struct ArgumentDeserialize<ExecutionSpace::Host, ArrayArgument<const T, L>> {
    static basic_view<const T, L, views::accessors::host> unpack(
        const TaskContext& context,
        ArrayArgument<const T, L> array
    ) {
        const T* data = static_cast<const T*>(context.accessors.at(array.buffer_index).address);

        return {data, array.layout};
    }
};

template<typename T, typename L>
struct ArgumentDeserialize<ExecutionSpace::Cuda, ArrayArgument<T, L>> {
    static basic_view<T, L, views::accessors::cuda_device> unpack(
        const TaskContext& context,
        ArrayArgument<T, L> array
    ) {
        T* data = static_cast<T*>(context.accessors.at(array.buffer_index).address);

        return {data, array.layout};
    }
};

template<typename T, typename L>
struct ArgumentDeserialize<ExecutionSpace::Cuda, ArrayArgument<const T, L>> {
    static basic_view<const T, L, views::accessors::cuda_device> unpack(
        const TaskContext& context,
        ArrayArgument<const T, L> array
    ) {
        const T* data = static_cast<const T*>(context.accessors.at(array.buffer_index).address);

        return {data, array.layout};
    }
};

}  // namespace kmm