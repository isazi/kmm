#pragma once

#include "kmm/core/reduction.hpp"
#include "kmm/utils/small_vector.hpp"

namespace kmm {

struct CopyDef {
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

struct FillDef {
    FillDef(size_t element_length, size_t num_elements, const void* fill_value) :
        offset_elements(0),
        num_elements(num_elements) {
        this->fill_value.insert_all(
            reinterpret_cast<const uint8_t*>(fill_value),
            reinterpret_cast<const uint8_t*>(fill_value) + element_length
        );
    }

    template<typename T>
    static FillDef with_value(const T& value, size_t num_elements = 1) {
        return {sizeof(T), num_elements, &value};
    }

    size_t minimum_destination_bytes_needed() const;

    size_t offset_elements = 0;
    size_t num_elements;
    small_buffer fill_value;
};

struct ReductionDef {
    size_t minimum_source_bytes_needed() const;
    size_t minimum_destination_bytes_needed() const;

    Reduction operation;
    DataType data_type;
    size_t num_outputs;
    size_t num_inputs_per_output = 1;
    size_t input_stride_elements = num_outputs;
    size_t input_offset_elements = 0;
    size_t output_offset_elements = 0;
};

}  // namespace kmm