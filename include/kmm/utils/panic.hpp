#pragma once

#include <iostream>

#define KMM_PANIC(...)                                           \
    do {                                                         \
        kmm::panic_format_args(__FILE__, __LINE__, __VA_ARGS__); \
        while (1)                                                \
            ;                                                    \
    } while (0);

#define KMM_ASSERT(...)                                   \
    do {                                                  \
        if (!static_cast<bool>(__VA_ARGS__)) {            \
            KMM_PANIC("assertion failed: " #__VA_ARGS__); \
        }                                                 \
    } while (0);

#define KMM_TODO()            KMM_PANIC("not implemented")
#define KMM_DEBUG_ASSERT(...) KMM_ASSERT(__VA_ARGS__)

namespace kmm {

[[noreturn]] void panic(const char* message);

using panic_formatter_fn = void (*)(std::ostream&, const void**);

[[noreturn]] void panic_format(
    const char* filename,
    int line,
    panic_formatter_fn formatter,
    const void** data);

template<typename... Args>
[[noreturn]] __attribute__((noinline)) void panic_format_args(
    const char* filename,
    int line,
    const Args&... args) {
    const void* data[] = {static_cast<const void*>(&args)...};
    auto formatter = [](std::ostream& stream, const void** args) {
        size_t index = 0;
        ((stream << static_cast<const Args*>(args[index++])), ...);
    };

    panic_format(filename, line, formatter, data);
}

}  // namespace kmm