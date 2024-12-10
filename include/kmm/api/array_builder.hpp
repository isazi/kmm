#pragma once

#include "kmm/core/buffer.hpp"
#include "kmm/core/geometry.hpp"
#include "kmm/core/reduction.hpp"
#include "kmm/core/view.hpp"

namespace kmm {

class TaskGraph;

template<size_t N>
struct DataChunk {
    BufferId buffer_id;
    MemoryId owner_id;
    Index<N> offset;
    Size<N> size;
};

template<size_t N>
class DataDistribution {
  public:
    DataDistribution(Size<N> array_size, std::vector<DataChunk<N>> chunks);

    DataChunk<N> find_chunk(Range<N> region) const;
    DataChunk<N> chunk(size_t index) const;

    size_t num_chunks() const {
        return m_chunks.size();
    }

    const std::vector<DataChunk<N>>& chunks() const {
        return m_chunks;
    }

    Size<N> chunk_size() const {
        return m_chunk_size;
    }

    Size<N> array_size() const {
        return m_array_size;
    }

  protected:
    std::vector<DataChunk<N>> m_chunks;
    std::array<size_t, N> m_chunks_count;
    Size<N> m_array_size = Size<N>::zero();
    Size<N> m_chunk_size = Size<N>::zero();
};

template<size_t N>
class ArrayBuilder {
  public:
    ArrayBuilder(Size<N> sizes, BufferLayout element_layout) :
        m_sizes(sizes),
        m_element_layout(element_layout) {}

    BufferRequirement add_chunk(TaskGraph& graph, MemoryId memory_id, Range<N> access_region);
    DataDistribution<N> build(TaskGraph& graph);

    Size<N> sizes() const {
        return m_sizes;
    }

  private:
    Size<N> m_sizes;
    BufferLayout m_element_layout;
    std::vector<DataChunk<N>> m_chunks;
};

template<size_t N>
class ArrayReductionBuilder {
  public:
    ArrayReductionBuilder(Size<N> sizes, DataType data_type, ReductionOp operation) :
        m_sizes(sizes),
        m_dtype(data_type),
        m_reduction(operation) {}

    BufferRequirement add_chunk(
        TaskGraph& graph,
        MemoryId memory_id,
        Range<N> access_region,
        size_t replication_factor = 1
    );

    DataDistribution<N> build(TaskGraph& graph);

    Size<N> sizes() const {
        return m_sizes;
    }

  private:
    Size<N> m_sizes;
    DataType m_dtype;
    ReductionOp m_reduction;
    std::unordered_map<Range<N>, std::vector<ReductionInput>> m_partial_inputs;
};

}  // namespace kmm