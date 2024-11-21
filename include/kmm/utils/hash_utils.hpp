#pragma once

#include <functional>

namespace kmm {

template<class T>
void hash_combine(size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template<typename It>
void hash_combine(size_t& seed, It begin, It end) {
    for (It current = begin; current != end; current++) {
        hash_combine(seed, *current);
    }
}

template<typename It>
size_t hash_range(It begin, It end) {
    size_t seed = 0;
    hash_combine(seed, begin, end);
    return seed;
}

}  // namespace kmm