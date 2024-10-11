#include <cstdint>
#include <cstring>
#include <functional>

#include "kmm/core/reduction.hpp"
#include "kmm/memops/host_copy.hpp"

namespace kmm {

inline void execute_copy_impl(
    const void* src_buffer,
    void* dst_buffer,
    CopyDescription copy_description
) {
    for (size_t i3 = 0; i3 < copy_description.counts[3]; i3++) {
        for (size_t i2 = 0; i2 < copy_description.counts[2]; i2++) {
            for (size_t i1 = 0; i1 < copy_description.counts[1]; i1++) {
                for (size_t i0 = 0; i0 < copy_description.counts[0]; i0++) {
                    size_t src_offset = copy_description.src_offset
                        + i0 * copy_description.src_strides[0]
                        + i1 * copy_description.src_strides[1]
                        + i2 * copy_description.src_strides[2]
                        + i3 * copy_description.src_strides[3];

                    size_t dst_offset = copy_description.dst_offset
                        + i0 * copy_description.dst_strides[0]
                        + i1 * copy_description.dst_strides[1]
                        + i2 * copy_description.dst_strides[2]
                        + i3 * copy_description.dst_strides[3];

                    ::memcpy(
                        static_cast<uint8_t*>(dst_buffer) + dst_offset,
                        static_cast<const uint8_t*>(src_buffer) + src_offset,
                        copy_description.element_size
                    );
                }
            }
        }
    }
}

template<size_t Align>
bool is_aligned(const void* src_buffer, void* dst_buffer, CopyDescription copy_description) {
    bool result = reinterpret_cast<uintptr_t>(src_buffer) % Align == 0
        && reinterpret_cast<uintptr_t>(dst_buffer) % Align == 0
        && copy_description.src_offset % Align == 0 && copy_description.dst_offset % Align == 0
        && copy_description.element_size % Align == 0;

    for (size_t i = 0; i < CopyDescription::MAX_DIMS; i++) {
        if (copy_description.counts[i] > 1) {
            result &= copy_description.src_strides[i] % Align == 0;
            result &= copy_description.dst_strides[i] % Align == 0;
        }
    }

    return result;
}

void execute_copy(const void* src_buffer, void* dst_buffer, CopyDescription copy_description) {
    copy_description.simplify();
    return execute_copy_impl(src_buffer, dst_buffer, copy_description);
}

}  // namespace kmm