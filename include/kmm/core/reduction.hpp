#pragma once

#include "kmm/core/data_type.hpp"
#include "kmm/core/identifiers.hpp"

namespace kmm {

enum struct Reduction : uint8_t { Sum, Product, Min, Max, BitAnd, BitOr };

std::vector<uint8_t> reduction_identity_value(DataType dtype, Reduction op);

struct ReductionInput {
    BufferId buffer_id;
    MemoryId memory_id;
    EventList dependencies;
    size_t num_inputs_per_output = 1;
};

struct ReductionOutput {
    Reduction operation;
    DataType data_type;
    size_t num_outputs;
};

std::ostream& operator<<(std::ostream& f, Reduction p);
std::ostream& operator<<(std::ostream& f, ReductionInput p);
std::ostream& operator<<(std::ostream& f, ReductionOutput p);

}  // namespace kmm

template<>
struct fmt::formatter<kmm::Reduction>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::ReductionInput>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::ReductionOutput>: fmt::ostream_formatter {};