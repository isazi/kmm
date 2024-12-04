#pragma once

#include "argument.hpp"

#include "kmm/core/geometry.hpp"
#include "kmm/core/identifiers.hpp"

namespace kmm {

template<size_t N>
struct ArrayChunk {
    BufferId buffer_id;
    MemoryId owner_id;
    Index<N> offset;
    Size<N> size;
};

template<size_t N>
class ArrayBackend: public std::enable_shared_from_this<ArrayBackend<N>> {
  public:
    ArrayBackend(
        std::shared_ptr<Worker> worker,
        Size<N> array_size,
        std::vector<ArrayChunk<N>> chunks
    );
    ~ArrayBackend();

    ArrayChunk<N> find_chunk(Range<N> region) const;
    ArrayChunk<N> chunk(size_t index) const;
    void synchronize() const;
    void copy_bytes(void* dest_addr, size_t element_size) const;

    size_t num_chunks() const {
        return m_buffers.size();
    }

    const std::vector<BufferId>& buffers() const {
        return m_buffers;
    }

    const Worker& worker() const {
        return *m_worker;
    }

    Size<N> chunk_size() const {
        return m_chunk_size;
    }

    Size<N> array_size() const {
        return m_array_size;
    }

  private:
    std::shared_ptr<Worker> m_worker;
    std::vector<BufferId> m_buffers;
    Size<N> m_array_size = Size<N>::zero();
    Size<N> m_chunk_size = Size<N>::zero();
    std::array<size_t, N> m_num_chunks;
};

}  // namespace kmm