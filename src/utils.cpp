#include "kmm/utils.hpp"

#include <cstdio>

namespace kmm {
__attribute__((noinline)) void panic(const char* filename, int line, const char* expression) {
    fprintf(stderr, "panic at %s:%d: %s\n", filename, line, expression);
    fflush(stderr);
    std::terminate();
}
}  // namespace kmm