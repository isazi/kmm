#pragma once

#include "array_backend.hpp"

namespace kmm {

class Worker;
class TaskGraph;
class ReductionInput;

template<size_t N>
struct ArrayBuilder {
    ArrayBuilder(Dim<N> sizes, BufferLayout element_layout) :
        m_sizes(sizes),
        m_element_layout(element_layout) {}

    size_t add_chunk(TaskBuilder& builder, Range<N> access_region);
    std::shared_ptr<ArrayBackend<N>> build(std::shared_ptr<Worker> worker, TaskGraph& graph);

    Dim<N> m_sizes;
    BufferLayout m_element_layout;
    std::vector<ArrayChunk<N>> m_chunks;
};

template<size_t N>
struct ArrayReductionBuilder {
    ArrayReductionBuilder(Dim<N> sizes, DataType data_type, ReductionOp operation) :
        m_sizes(sizes),
        m_dtype(data_type),
        m_reduction(operation) {}

    size_t add_chunk(TaskBuilder& builder, Range<N> access_region, size_t replication_factor = 1);
    std::shared_ptr<ArrayBackend<N>> build(std::shared_ptr<Worker> worker, TaskGraph& graph);

    Dim<N> m_sizes;
    DataType m_dtype;
    ReductionOp m_reduction;
    std::unordered_map<Range<N>, std::vector<ReductionInput>> m_partial_inputs;
};

}  // namespace kmm