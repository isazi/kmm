#pragma once

#include <functional>
#include <variant>
#include <vector>

#define KMM_PANIC(...)                                  \
    do {                                                \
        ::kmm::panic(__FILE__, __LINE__, #__VA_ARGS__); \
        while (1)                                       \
            ;                                           \
    } while (0)

#define KMM_TODO() KMM_PANIC("not implemented")

#define KMM_ASSERT(...)                        \
    do {                                       \
        if (!static_cast<bool>(__VA_ARGS__)) { \
            KMM_PANIC(#__VA_ARGS__);           \
        }                                      \
    } while (0)

#define KMM_DEBUG_ASSERT(...) KMM_ASSERT(__VA_ARGS__)

namespace kmm {

template<typename T>
void remove_duplicates(T& input) {
    std::sort(std::begin(input), std::end(input));
    auto last_unique = std::unique(std::begin(input), std::end(input));
    input.erase(last_unique, std::end(input));
}

[[noreturn]] __attribute__((noinline)) void panic(
    const char* filename,
    int line,
    const char* expression);

template<typename T>
T checked_product(const T* begin, const T* end) {
    T result = 1;
    bool overflowed = false;

    for (auto it = begin; it != end; it++) {
        overflowed |= __builtin_mul_overflow(*it, result, &result);
    }

    KMM_ASSERT(overflowed == false);
    return result;
}

}  // namespace kmm
