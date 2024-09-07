#include "spdlog/spdlog.h"

#include "kmm/api/array.hpp"
#include "kmm/api/runtime.hpp"
#include "kmm/internals/worker.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

template<size_t N>
ArrayBackend<N>::ArrayBackend(
    std::shared_ptr<Worker> worker,
    dim<N> array_size,
    std::vector<ArrayChunk<N>> chunks) :
    m_worker(worker),
    m_array_size(array_size) {
    for (const auto& chunk : chunks) {
        if (chunk.offset == point<N>::zero()) {
            m_chunk_size = chunk.size;
        }
    }

    if (m_chunk_size.is_empty()) {
        throw std::runtime_error("chunk size cannot be empty");
    }

    size_t num_total_chunks = 1;

    for (size_t i = 0; i < N; i++) {
        m_num_chunks[i] = checked_cast<size_t>(div_ceil(array_size[i], m_chunk_size[i]));
        num_total_chunks *= m_num_chunks[i];
    }

    static constexpr size_t invalid_index = static_cast<size_t>(-1);
    std::vector<size_t> buffer_locs(num_total_chunks, invalid_index);

    for (const auto& chunk : chunks) {
        size_t buffer_index = 0;

        for (size_t i = 0; i < N; i++) {
            auto k = div_floor(chunk.offset[i], m_chunk_size[i]);
            auto i0 = k * m_chunk_size[i];
            auto i1 = std::min(m_chunk_size[i], array_size[i] - i0);

            // check if in bounds
            if (chunk.offset[i] != i0 || chunk.size[i] != i1) {
                KMM_PANIC("out of bounds");
            }

            buffer_index = buffer_index * m_num_chunks[i] + static_cast<size_t>(k);
        }

        if (buffer_locs[buffer_index] != invalid_index) {
            KMM_TODO();
        }

        buffer_locs[buffer_index] = buffer_index;
    }

    for (size_t index : buffer_locs) {
        if (index == invalid_index) {
            KMM_TODO();
        }
    }

    for (size_t index : buffer_locs) {
        m_buffers.push_back(chunks[index].buffer_id);
    }
}

template<size_t N>
ArrayBackend<N>::~ArrayBackend() {
    m_worker->with_task_graph([&](auto& builder) {
        for (auto id : m_buffers) {
            builder.delete_buffer(id);
        }
    });
}

template<size_t N>
ArrayChunk<N> ArrayBackend<N>::find_chunk(rect<N> region) const {
    size_t buffer_index = 0;
    point<N> offset;
    dim<N> sizes;

    for (size_t i = 0; i < N; i++) {
        auto k = region.offset[i] / m_chunk_size[i];
        auto w = region.offset[i] % m_chunk_size[i] + region.sizes[i];

        if (!in_range(k, m_num_chunks[i]) || w > m_chunk_size[i]) {
            throw std::out_of_range("invalid chunk");
        }

        buffer_index = buffer_index * m_num_chunks[i] + static_cast<size_t>(k);
        offset[i] = k * m_chunk_size[i];
        sizes[i] = m_chunk_size[i];
    }

    // TODO
    MemoryId owner_id = MemoryId::host();

    spdlog::debug("find_chunk {} -> {} (index: {})", region, rect<N>(offset, sizes), buffer_index);
    return {m_buffers[buffer_index], owner_id, offset, sizes};
}

template<size_t N>
ArrayChunk<N> ArrayBackend<N>::chunk(size_t index) const {
    if (index >= m_buffers.size()) {
        throw std::runtime_error(
            fmt::format("index {} is out of range for array of size {}", index, m_buffers.size()));
    }

    point<N> offset;
    dim<N> size;

    for (size_t j = 0; j < N; j++) {
        size_t i = N - 1 - j;
        auto k = index % m_num_chunks[i];
        index /= m_num_chunks[i];

        offset[i] = int64_t(k) * m_chunk_size[i];
        size[i] = std::min(m_chunk_size[i], m_array_size[i] - offset[i]);
    }

    // TODO
    MemoryId owner_id = MemoryId::host();

    return {m_buffers[index], owner_id, offset, size};
}

template<size_t N>
void ArrayBackend<N>::synchronize() const {
    auto event_id = m_worker->with_task_graph([&](TaskGraph& graph) {
        auto deps = EventList {};

        for (const auto& buffer_id : m_buffers) {
            graph.access_buffer(buffer_id, AccessMode::ReadWrite, deps);
        }

        return graph.join_events(deps);
    });

    m_worker->query_event(event_id, std::chrono::system_clock::time_point::max());
}

template class ArrayBackend<0>;
template class ArrayBackend<1>;
template class ArrayBackend<2>;
template class ArrayBackend<3>;
template class ArrayBackend<4>;
template class ArrayBackend<5>;
template class ArrayBackend<6>;

}  // namespace kmm