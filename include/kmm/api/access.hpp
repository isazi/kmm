#pragma once
#include "mapper.hpp"

namespace kmm {
template<typename T, typename I = All>
struct Read {
    T argument;
    I access_map;
};

template<typename T, typename I = All>
struct Write {
    T& argument;
    I access_map;
};

template<typename T, typename I = All, typename R = MultiIndexMap<0>>
struct Reduce {
    T& argument;
    ReductionOp op;
    I access_map = {};
    R private_map = {};
};

template<typename I = All, typename T>
Read<T, I> read(T argument, I access_map = {}) {
    return {argument, access_map};
}

template<typename I = All, typename T>
Write<T, I> write(T& argument, I access_map = {}) {
    return {argument, access_map};
}

template<typename I>
struct Privatize {
    I access_map;
};

template<typename... Is>
Privatize<MultiIndexMap<sizeof...(Is)>> privatize(const Is&... slices) {
    return {into_index_map(slices)...};
}

template<typename T>
Reduce<T> reduce(T& argument, ReductionOp op) {
    return {argument, op};
}

template<typename T, typename I>
Reduce<T, I> reduce(T& argument, ReductionOp op, I access_map) {
    return {argument, op, access_map};
}

template<typename T, typename I, typename P>
Reduce<T, I, P> reduce(T& argument, ReductionOp op, Privatize<P> private_map, I access_map) {
    return {argument, op, access_map, private_map.access_map};
}

template<typename T, typename P>
Reduce<T, All, P> reduce(T& argument, ReductionOp op, Privatize<P> private_map) {
    return {argument, op, All {}, private_map.access_map};
}
}  // namespace kmm