#pragma once

#include "kmm/utils/small_vector.hpp"

namespace kmm {

class FillDef {
  public:
    FillDef(size_t element_length, size_t num_elements, const void* fill_value) :
        offset_elements(0),
        num_elements(num_elements) {
        this->fill_value.insert_all(
            reinterpret_cast<const uint8_t*>(fill_value),
            reinterpret_cast<const uint8_t*>(fill_value) + element_length
        );
    }

    //  public:
    size_t offset_elements = 0;
    size_t num_elements;
    small_vector<uint8_t, sizeof(uint64_t)> fill_value;
};

}  // namespace kmm