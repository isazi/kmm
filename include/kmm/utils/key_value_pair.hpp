#pragma once

#include "kmm/core/macros.hpp"

namespace kmm {

template<typename V>
struct alignas(2 * alignof(int64_t)) KeyValue {
    V value;
    int64_t key;
};

template<typename T>
KMM_HOST_DEVICE bool operator==(const KeyValue<T>& a, const KeyValue<T>& b) {
    return a.value == b.value && a.key == b.key;
}

template<typename T>
KMM_HOST_DEVICE bool operator<(const KeyValue<T>& a, const KeyValue<T>& b) {
    return a.value < b.value || (!(a.value > b.value) && a.key < b.key);
}

template<typename T>
KMM_HOST_DEVICE bool operator<=(const KeyValue<T>& a, const KeyValue<T>& b) {
    return a.value < b.value || (!(a.value > b.value) && a.key <= b.key);
}

template<typename T>
KMM_HOST_DEVICE bool operator!=(const KeyValue<T>& a, const KeyValue<T>& b) {
    return !(a == b);
}

template<typename T>
KMM_HOST_DEVICE bool operator>(const KeyValue<T>& a, const KeyValue<T>& b) {
    return b < a;
}

template<typename T>
KMM_HOST_DEVICE bool operator>=(const KeyValue<T>& a, const KeyValue<T>& b) {
    return b <= a;
}

}  // namespace kmm