#include "kmm/api/array_builder.hpp"
#include "kmm/internals/worker.hpp"

namespace kmm {

template<size_t N>
BufferRequirement ArrayBuilder<N>::add_chunk(
    TaskGraph& graph,
    MemoryId memory_id,
    Range<N> access_region
) {
    auto num_elements = checked_cast<size_t>(access_region.size());
    auto buffer_id = graph.create_buffer(m_element_layout.repeat(num_elements));

    m_chunks.push_back(DataChunk<N> {
        .buffer_id = buffer_id,
        .owner_id = memory_id,
        .offset = access_region.offset,
        .size = access_region.sizes});

    return {.buffer_id = buffer_id, .memory_id = memory_id, .access_mode = AccessMode::Exclusive};
}

template<size_t N>
DataDistribution<N> ArrayBuilder<N>::build(TaskGraph& graph) {
    return DataDistribution<N>(m_sizes, std::move(m_chunks));
}

BufferLayout make_layout(size_t num_elements, DataType dtype, ReductionOp reduction) {
    return BufferLayout {
        .size_in_bytes = dtype.size_in_bytes() * num_elements,
        .alignment = dtype.alignment(),
        .fill_pattern = reduction_identity_value(dtype, reduction),
    };
}

template<size_t N>
BufferRequirement ArrayReductionBuilder<N>::add_chunk(
    TaskGraph& graph,
    MemoryId memory_id,
    Range<N> access_region,
    size_t replication_factor
) {
    auto num_elements = checked_mul(checked_cast<size_t>(access_region.size()), replication_factor);
    auto buffer_id = graph.create_buffer(make_layout(num_elements, m_dtype, m_reduction));

    m_partial_inputs[access_region].push_back(ReductionInput {
        .buffer_id = buffer_id,
        .memory_id = memory_id,
        .dependencies = {},
        .num_inputs_per_output = replication_factor});

    return {.buffer_id = buffer_id, .memory_id = memory_id, .access_mode = AccessMode::Exclusive};
}

template<size_t N>
DataDistribution<N> ArrayReductionBuilder<N>::build(TaskGraph& graph) {
    std::vector<DataChunk<N>> chunks;

    for (auto& p : m_partial_inputs) {
        auto access_region = p.first;
        auto& inputs = p.second;

        MemoryId memory_id = inputs[0].memory_id;
        auto num_elements = checked_cast<size_t>(access_region.size());

        auto buffer_id = graph.create_buffer(BufferLayout::for_type(m_dtype).repeat(num_elements));

        auto reduction = Reduction {
            .operation = m_reduction,  //
            .data_type = m_dtype,
            .num_outputs = num_elements};

        auto event_id = graph.insert_multilevel_reduction(buffer_id, memory_id, reduction, inputs);

        chunks.push_back(DataChunk<N> {
            .buffer_id = buffer_id,
            .owner_id = memory_id,
            .offset = access_region.offset,
            .size = access_region.sizes});

        for (const auto& input : inputs) {
            graph.delete_buffer(input.buffer_id, {event_id});
        }
    }

    return DataDistribution<N>(m_sizes, std::move(chunks));
}

// NOLINTBEGIN
#define INSTANTIATE_ARRAY_IMPL(NAME) \
    template class NAME<0>;          \
    template class NAME<1>;          \
    template class NAME<2>;          \
    template class NAME<3>;          \
    template class NAME<4>;          \
    template class NAME<5>;          \
    template class NAME<6>;

INSTANTIATE_ARRAY_IMPL(ArrayBuilder)
INSTANTIATE_ARRAY_IMPL(ArrayReductionBuilder)
// NOLINTEND

}  // namespace kmm