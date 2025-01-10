#pragma once

#include "kmm/api/argument.hpp"
#include "kmm/utils/view.hpp"

namespace kmm {

template<typename T, typename D, typename L = views::default_layout<D::rank>>
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
struct ArgumentUnpack<ExecutionSpace::Host, ArrayArgument<T, D, L>> {
    using type = basic_view<T, D, L, views::host_accessor>;

    static type call(const TaskContext& context, ArrayArgument<T, D, L> arg) {
        T* data = static_cast<T*>(context.accessors.at(arg.buffer_index).address);
        return {data, arg.domain, arg.layout};
    }
};

template<typename T, typename D, typename L>
struct ArgumentUnpack<ExecutionSpace::Host, ArrayArgument<const T, D, L>> {
    using type = basic_view<const T, D, L, views::host_accessor>;

    static type call(const TaskContext& context, ArrayArgument<const T, D, L> arg) {
        const T* data = static_cast<const T*>(context.accessors.at(arg.buffer_index).address);
        return {data, arg.domain, arg.layout};
    }
};

template<typename T, typename D, typename L>
struct ArgumentUnpack<ExecutionSpace::Device, ArrayArgument<T, D, L>> {
    using type = basic_view<T, D, L, views::device_accessor>;

    static type call(const TaskContext& context, ArrayArgument<T, D, L> arg) {
        T* data = static_cast<T*>(context.accessors.at(arg.buffer_index).address);
        return {data, arg.domain, arg.layout};
    }
};

template<typename T, typename D, typename L>
struct ArgumentUnpack<ExecutionSpace::Device, ArrayArgument<const T, D, L>> {
    using type = basic_view<const T, D, L, views::device_accessor>;

    static type call(const TaskContext& context, ArrayArgument<const T, D, L> arg) {
        const T* data = static_cast<const T*>(context.accessors.at(arg.buffer_index).address);
        return {data, arg.domain, arg.layout};
    }
};

}  // namespace kmm