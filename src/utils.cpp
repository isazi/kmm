#include "kmm/utils.hpp"

#include <cstdio>

namespace kmm {
void panic(const char* filename, int line, const char* expression) {
    fprintf(stderr, "assertion failed (%s:%d): %s\n", filename, line, expression);
    std::terminate();
}
}  // namespace kmm