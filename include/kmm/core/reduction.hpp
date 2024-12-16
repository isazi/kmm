#pragma once

#include "kmm/core/data_type.hpp"
#include "kmm/core/identifiers.hpp"

namespace kmm {

enum struct ReductionOp : uint8_t { Sum, Product, Min, Max, BitAnd, BitOr };

std::vector<uint8_t> reduction_identity_value(DataType dtype, ReductionOp op);

struct ReductionDef {
    ReductionOp operation;
    DataType data_type;
    size_t num_outputs;
    size_t num_inputs_per_output = 1;
    size_t input_stride_elements = num_inputs_per_output;
    size_t input_offset_elements = 0;
    size_t output_offset_elements = 0;

    size_t minimum_source_bytes_needed() const;
    size_t minimum_destination_bytes_needed() const;
};

struct Reduction {
    ReductionOp operation;
    DataType data_type;
    size_t num_outputs;
};

struct ReductionInput {
    BufferId buffer_id;
    MemoryId memory_id;
    EventList dependencies;
    size_t num_inputs_per_output = 1;
};

std::ostream& operator<<(std::ostream& f, ReductionOp p);
std::ostream& operator<<(std::ostream& f, Reduction p);
std::ostream& operator<<(std::ostream& f, ReductionDef p);
std::ostream& operator<<(std::ostream& f, ReductionInput p);

}  // namespace kmm

template<>
struct fmt::formatter<kmm::ReductionOp>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::Reduction>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::ReductionDef>: fmt::ostream_formatter {};

template<>
struct fmt::formatter<kmm::ReductionInput>: fmt::ostream_formatter {};