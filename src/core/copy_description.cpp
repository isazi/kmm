#include <cstddef>
#include <stdexcept>
#include <utility>

#include "kmm/core/copy_description.hpp"
#include "kmm/utils/checked_math.hpp"

namespace kmm {

size_t CopyDescription::minimum_source_bytes_needed() const {
    size_t result = src_offset;

    for (size_t i = 0; i < MAX_DIMS; i++) {
        result += checked_mul(counts[i], src_strides[i]);
    }

    return result + element_size;
}

size_t CopyDescription::minimum_destination_bytes_needed() const {
    size_t result = dst_offset;

    for (size_t i = 0; i < MAX_DIMS; i++) {
        result += checked_mul(counts[i], dst_strides[i]);
    }

    return result + element_size;
}

void CopyDescription::add_dimension(
    size_t count,
    size_t src_offset,
    size_t dst_offset,
    size_t src_stride,
    size_t dst_stride) {
    for (size_t i = 0; i < MAX_DIMS; i++) {
        if (counts[i] == 1) {
            this->src_offset += src_offset * src_stride * element_size;
            this->dst_offset += dst_offset * dst_stride * element_size;

            counts[i] = count;
            src_strides[i] = src_stride * element_size;
            dst_strides[i] = dst_stride * element_size;
            return;
        }
    }

    throw std::length_error("the number of dimensions of a copy operation cannot exceed 4");
}

size_t CopyDescription::effective_dimensionality() const {
    for (size_t n = MAX_DIMS; n > 0; n--) {
        if (counts[n - 1] != 1) {
            return n;
        }
    }

    return 0;
}

size_t CopyDescription::number_of_bytes_copied() const {
    return checked_mul(checked_product(counts, counts + MAX_DIMS), element_size);
}

void CopyDescription::simplify() {
    if (number_of_bytes_copied() == 0) {
        element_size = 0;
        src_offset = 0;
        dst_offset = 0;

        for (size_t i = 0; i < MAX_DIMS; i++) {
            counts[i] = 0;
            src_strides[i] = 1;
            dst_strides[i] = 1;
        }

        return;
    }

    for (size_t i = 0; i < MAX_DIMS; i++) {
        for (size_t j = 0; j < MAX_DIMS; j++) {
            if (src_strides[j] == element_size && dst_strides[j] == element_size) {
                element_size *= counts[j];
                counts[j] = 1;
                src_strides[j] = 1;
                dst_strides[j] = 1;
            }
        }
    }

    for (size_t i = 0; i < MAX_DIMS; i++) {
        for (size_t j = 0; j < MAX_DIMS; j++) {
            if (i != j && src_strides[j] == counts[i] * src_strides[i]
                && dst_strides[j] == counts[i] * dst_strides[i]) {
                counts[i] *= counts[j];

                counts[j] = 1;
                src_strides[j] = 1;
                dst_strides[j] = 1;
            }
        }
    }

    for (size_t i = 0; i < MAX_DIMS; i++) {
        if (counts[i] == 1) {
            src_strides[i] = 1;
            dst_strides[i] = 1;
        }
    }

    for (size_t i = 0; i < MAX_DIMS; i++) {
        for (size_t j = i + 1; j < MAX_DIMS; j++) {
            if ((counts[i] == 1 && counts[j] != 1) || dst_strides[i] > dst_strides[j]
                || (dst_strides[i] == dst_strides[j] && src_strides[i] > src_strides[j])) {
                std::swap(counts[i], counts[j]);
                std::swap(src_strides[i], src_strides[j]);
                std::swap(dst_strides[i], dst_strides[j]);
            }
        }
    }

    for (size_t i = 0; i < MAX_DIMS; i++) {
        if (counts[i] == 1) {
            if (i == 0) {
                src_strides[0] = element_size;
                dst_strides[0] = element_size;
            } else {
                src_strides[i] = src_strides[i - 1];
                dst_strides[i] = dst_strides[i - 1];
            }
        }
    }
}

}  // namespace kmm