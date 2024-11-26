#include "kmm/api/array_builder.hpp"
#include "kmm/internals/worker.hpp"

namespace kmm {

template<size_t N>
size_t ArrayBuilder<N>::add_chunk(TaskBuilder& builder, Range<N> access_region) {
    auto num_elements = checked_cast<size_t>(access_region.size());
    auto buffer_id = builder.graph.create_buffer(m_element_layout.repeat(num_elements));

    m_chunks.push_back(ArrayChunk<N> {
        .buffer_id = buffer_id,
        .owner_id = builder.memory_id,
        .offset = access_region.offset,
        .size = access_region.sizes});

    size_t buffer_index = builder.buffers.size();
    builder.buffers.emplace_back(BufferRequirement {
        .buffer_id = buffer_id,
        .memory_id = builder.memory_id,
        .access_mode = AccessMode::Exclusive});

    return buffer_index;
}

template<size_t N>
std::shared_ptr<ArrayBackend<N>> ArrayBuilder<N>::build(
    std::shared_ptr<Worker> worker,
    TaskGraph& graph
) {
    return std::make_shared<ArrayBackend<N>>(worker, m_sizes, std::move(m_chunks));
}

BufferLayout make_layout(size_t num_elements, DataType dtype, ReductionOp reduction) {
    return BufferLayout {
        .size_in_bytes = dtype.size_in_bytes() * num_elements,
        .alignment = dtype.alignment(),
        .fill_pattern = reduction_identity_value(dtype, reduction),
    };
}

template<size_t N>
size_t ArrayReductionBuilder<N>::add_chunk(
    TaskBuilder& builder,
    Range<N> access_region,
    size_t replication_factor
) {
    auto num_elements = checked_mul(checked_cast<size_t>(access_region.size()), replication_factor);
    auto memory_id = builder.memory_id;

    auto buffer_id = builder.graph.create_buffer(make_layout(num_elements, m_dtype, m_reduction));

    m_partial_inputs[access_region].push_back(ReductionInput {
        .buffer_id = buffer_id,
        .memory_id = memory_id,
        .dependencies = {},
        .num_inputs_per_output = replication_factor});

    size_t buffer_index = builder.buffers.size();
    builder.buffers.emplace_back(BufferRequirement {
        .buffer_id = buffer_id,
        .memory_id = memory_id,
        .access_mode = AccessMode::Exclusive});

    return buffer_index;
}

template<size_t N>
std::shared_ptr<ArrayBackend<N>> ArrayReductionBuilder<N>::build(
    std::shared_ptr<Worker> worker,
    TaskGraph& graph
) {
    std::vector<ArrayChunk<N>> chunks;

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

        chunks.push_back(ArrayChunk<N> {
            .buffer_id = buffer_id,
            .owner_id = memory_id,
            .offset = access_region.offset,
            .size = access_region.sizes});

        for (const auto& input : inputs) {
            graph.delete_buffer(input.buffer_id, {event_id});
        }
    }

    return std::make_shared<ArrayBackend<N>>(worker, m_sizes, std::move(chunks));
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