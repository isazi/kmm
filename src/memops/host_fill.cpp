#include <cstdint>

#include "kmm/memops/host_fill.hpp"

namespace kmm {

void execute_fill(void* dst_buffer, size_t nbytes, const void* pattern, size_t pattern_nbytes) {
    // TODO: optimize
    for (size_t i = 0; i < nbytes; i++) {
        static_cast<uint8_t*>(dst_buffer)[i] =
            static_cast<const uint8_t*>(pattern)[i % pattern_nbytes];
    }
}

}  // namespace kmm