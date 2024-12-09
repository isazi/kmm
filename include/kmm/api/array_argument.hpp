#pragma once

#include "argument.hpp"

#include "kmm/core/view.hpp"

namespace kmm {

template<typename T, typename D, typename L = views::layout_default<D::rank>>
struct ArrayArgument {
    using value_type = T;
    using domain_type = D;
    using layout_type = L;

    ArrayArgument(size_t buffer_index, D domain, L layout) :
        buffer_index(buffer_index),
        domain(domain),
        layout(layout) {}

    ArrayArgument(size_t buffer_index, D domain) :
        ArrayArgument(buffer_index, domain, L::from_domain(domain)) {}

    size_t buffer_index;
    D domain;
    L layout;
};

template<typename T, typename D, typename L>
struct ArgumentDeserialize<ExecutionSpace::Host, ArrayArgument<T, D, L>> {
    using type = basic_view<T, D, L, views::accessor_host>;

    static type unpack(const TaskContext& context, ArrayArgument<T, D, L> arg) {
        T* data = static_cast<T*>(context.accessors.at(arg.buffer_index).address);
        return {data, arg.domain, arg.layout};
    }
};

template<typename T, typename D, typename L>
struct ArgumentDeserialize<ExecutionSpace::Host, ArrayArgument<const T, D, L>> {
    using type = basic_view<const T, D, L, views::accessor_host>;

    static type unpack(const TaskContext& context, ArrayArgument<const T, D, L> arg) {
        const T* data = static_cast<const T*>(context.accessors.at(arg.buffer_index).address);
        return {data, arg.domain, arg.layout};
    }
};

template<typename T, typename D, typename L>
struct ArgumentDeserialize<ExecutionSpace::Device, ArrayArgument<T, D, L>> {
    using type = basic_view<T, D, L, views::accessor_device>;

    static type unpack(const TaskContext& context, ArrayArgument<T, D, L> arg) {
        T* data = static_cast<T*>(context.accessors.at(arg.buffer_index).address);
        return {data, arg.domain, arg.layout};
    }
};

template<typename T, typename D, typename L>
struct ArgumentDeserialize<ExecutionSpace::Device, ArrayArgument<const T, D, L>> {
    using type = basic_view<const T, D, L, views::accessor_device>;

    static type unpack(const TaskContext& context, ArrayArgument<const T, D, L> arg) {
        const T* data = static_cast<const T*>(context.accessors.at(arg.buffer_index).address);
        return {data, arg.domain, arg.layout};
    }
};

}  // namespace kmm