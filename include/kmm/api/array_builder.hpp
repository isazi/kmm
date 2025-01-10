#pragma once

#include "kmm/core/buffer.hpp"
#include "kmm/core/geometry.hpp"
#include "kmm/core/reduction.hpp"
#include "kmm/utils/view.hpp"

namespace kmm {

class TaskGraph;

template<size_t N>
struct DataChunk {
    MemoryId owner_id;
    Index<N> offset;
    Size<N> size;
};

template<size_t N>
class DataDistribution {
  public:
    DataDistribution(Size<N> array_size, std::vector<DataChunk<N>> chunks);

    size_t region_to_chunk_index(Range<N> region) const;

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
    std::vector<size_t> m_mapping;
    std::array<size_t, N> m_chunks_count;
    Size<N> m_array_size = Size<N>::zero();
    Size<N> m_chunk_size = Size<N>::zero();
};

template<size_t N>
class ArrayBuilder {
  public:
    ArrayBuilder(Size<N> sizes, DataLayout element_layout) :
        m_sizes(sizes),
        m_element_layout(element_layout) {}

    BufferRequirement add_chunk(TaskGraph& graph, MemoryId memory_id, Range<N> access_region);
    std::pair<DataDistribution<N>, std::vector<BufferId>> build(TaskGraph& graph);

    Size<N> sizes() const {
        return m_sizes;
    }

  private:
    Size<N> m_sizes;
    DataLayout m_element_layout;
    std::vector<DataChunk<N>> m_chunks;
    std::vector<BufferId> m_buffers;
};

template<size_t N>
class ArrayReductionBuilder {
  public:
    ArrayReductionBuilder(Size<N> sizes, DataType data_type, Reduction operation) :
        m_sizes(sizes),
        m_dtype(data_type),
        m_reduction(operation) {}

    BufferRequirement add_chunk(
        TaskGraph& graph,
        MemoryId memory_id,
        Range<N> access_region,
        size_t replication_factor = 1
    );

    void add_chunks(ArrayReductionBuilder<N>&& other);

    std::pair<DataDistribution<N>, std::vector<BufferId>> build(TaskGraph& graph);

    Size<N> sizes() const {
        return m_sizes;
    }

  private:
    Size<N> m_sizes;
    DataType m_dtype;
    Reduction m_reduction;
    std::unordered_map<Range<N>, std::vector<ReductionInput>> m_partial_inputs;
};

}  // namespace kmm