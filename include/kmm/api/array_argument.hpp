#pragma once

#include "argument.hpp"

#include "kmm/core/view.hpp"

namespace kmm {

template<typename T, typename D>
struct ArrayArgument {
    size_t buffer_index;
    D domain;
};

template<typename T, typename D>
struct ArgumentDeserialize<ExecutionSpace::Host, ArrayArgument<T, D>> {
    using type = basic_view<T, views::layouts::right_to_left<D>, views::accessors::host>;

    static type unpack(const TaskContext& context, ArrayArgument<T, D> array) {
        T* data = static_cast<T*>(context.accessors.at(array.buffer_index).address);
        return {data, array.domain};
    }
};

template<typename T, typename D>
struct ArgumentDeserialize<ExecutionSpace::Host, ArrayArgument<const T, D>> {
    using type = basic_view<const T, views::layouts::right_to_left<D>, views::accessors::host>;

    static type unpack(const TaskContext& context, ArrayArgument<const T, D> array) {
        const T* data = static_cast<const T*>(context.accessors.at(array.buffer_index).address);
        return {data, array.domain};
    }
};

template<typename T, typename D>
struct ArgumentDeserialize<ExecutionSpace::Cuda, ArrayArgument<T, D>> {
    using type = basic_view<T, views::layouts::right_to_left<D>, views::accessors::cuda_device>;

    static type unpack(const TaskContext& context, ArrayArgument<T, D> array) {
        T* data = static_cast<T*>(context.accessors.at(array.buffer_index).address);
        return {data, array.domain};
    }
};

template<typename T, typename D>
struct ArgumentDeserialize<ExecutionSpace::Cuda, ArrayArgument<const T, D>> {
    using type =
        basic_view<const T, views::layouts::right_to_left<D>, views::accessors::cuda_device>;

    static type unpack(const TaskContext& context, ArrayArgument<const T, D> array) {
        const T* data = static_cast<const T*>(context.accessors.at(array.buffer_index).address);
        return {data, array.domain};
    }
};

}  // namespace kmm