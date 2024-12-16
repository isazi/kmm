#include <cstdint>

#include "kmm/memops/host_fill.hpp"

namespace kmm {

void execute_fill(void* dst_buffer, const FillDef& fill) {
    // TODO: optimize
    size_t k = fill.fill_value.size();

    for (size_t i = 0; i < fill.num_elements; i++) {
        for (size_t j = 0; j < k; j++) {
            static_cast<uint8_t*>(dst_buffer)[(fill.offset_elements + i) * k + j] =
                fill.fill_value[j];
        }
    }
}

}  // namespace kmm