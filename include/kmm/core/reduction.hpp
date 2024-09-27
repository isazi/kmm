#pragma once

#include "kmm/core/data_type.hpp"

namespace kmm {

enum struct ReductionOp : uint8_t { Sum, Product, Min, Max, BitAnd, BitOr };

std::vector<uint8_t> reduction_identity_value(DataType dtype, ReductionOp op);

struct Reduction {
    ReductionOp operation;
    DataType data_type;
    size_t num_outputs;
    size_t num_inputs_per_output = 1;
};

std::ostream& operator<<(std::ostream& f, ReductionOp p);
std::ostream& operator<<(std::ostream& f, Reduction p);

}  // namespace kmm

template<>
struct fmt::formatter<kmm::ReductionOp>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::Reduction>: fmt::ostream_formatter {};