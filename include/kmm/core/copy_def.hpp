#pragma once

#include "geometry.hpp"

namespace kmm {

class CopyDef {
  public:
    static constexpr size_t MAX_DIMS = 3;

    CopyDef(size_t element_size = 0) : element_size(element_size) {}

    void add_dimension(
        size_t count,
        size_t src_offset,
        size_t dst_offset,
        size_t src_stride,
        size_t dst_stride
    );

    size_t minimum_source_bytes_needed() const;
    size_t minimum_destination_bytes_needed() const;
    size_t number_of_bytes_copied() const;
    size_t effective_dimensionality() const;

    void simplify();

    size_t element_size = 0;
    size_t src_offset = 0;
    size_t dst_offset = 0;
    size_t counts[MAX_DIMS] = {1, 1, 1};
    size_t src_strides[MAX_DIMS] = {0, 0, 0};
    size_t dst_strides[MAX_DIMS] = {0, 0, 0};
};

}  // namespace kmm