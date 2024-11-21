#include <cstddef>

namespace kmm {

void execute_fill(void* dst_buffer, size_t nbytes, const void* pattern, size_t pattern_nbytes);

}