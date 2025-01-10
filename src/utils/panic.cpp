#include <execinfo.h>
#include <sstream>

#include "kmm/utils/panic.hpp"

namespace kmm {

void panic(const char* message) {
    fprintf(stderr, "fatal error: %s\n", message);
    exit(1);
}

void panic_format(const char* filename, int line, panic_formatter_fn formatter, const void** data) {
    std::stringstream stream;
    stream << filename << ":" << line << ": ";
    formatter(stream, data);

    panic(stream.str().c_str());
}

}  // namespace kmm