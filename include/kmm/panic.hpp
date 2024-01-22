#pragma once

#include <sstream>

#define KMM_PANIC(...)                                     \
    do {                                                   \
        ::kmm::panic_fmt(__FILE__, __LINE__, __VA_ARGS__); \
        while (1)                                          \
            ;                                              \
    } while (0)

#define KMM_ASSERT(...)                                   \
    do {                                                  \
        if (!static_cast<bool>(__VA_ARGS__)) {            \
            KMM_PANIC("assertion failed: " #__VA_ARGS__); \
        }                                                 \
    } while (0)

#define KMM_DEBUG_ASSERT(...) KMM_ASSERT(__VA_ARGS__)
#define KMM_TODO()            KMM_PANIC("not implemented")

namespace kmm {

[[noreturn]] void panic(const char* filename, int line, const char* expression);

template<typename... Args>
[[noreturn]] __attribute__((noinline)) void panic_fmt(
    const char* filename,
    int line,
    Args&&... args) {
    std::stringstream stream;
    ((stream << args), ...);
    panic(filename, line, stream.str().c_str());
}

}  // namespace kmm
