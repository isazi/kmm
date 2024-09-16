#include "spdlog/spdlog.h"

#include "kmm/api/array.hpp"
#include "kmm/internals/worker.hpp"
#include "kmm/memops/host_copy.hpp"
#include "kmm/utils/integer_fun.hpp"

namespace kmm {

template<size_t N>
rect<N> index2region(
    size_t index,
    std::array<size_t, N> num_chunks,
    dim<N> chunk_size,
    dim<N> array_size) {
    point<N> offset;
    dim<N> sizes;

    for (size_t j = 0; j < N; j++) {
        size_t i = N - 1 - j;
        auto k = index % num_chunks[i];
        index /= num_chunks[i];

        offset[i] = int64_t(k) * chunk_size[i];
        sizes[i] = std::min(chunk_size[i], array_size[i] - offset[i]);
    }

    return {offset, sizes};
}

template<size_t N>
ArrayBackend<N>::ArrayBackend(
    std::shared_ptr<Worker> worker,
    dim<N> array_size,
    dim<N> chunk_size,
    std::vector<ArrayChunk<N>> chunks) :
    m_worker(worker),
    m_chunk_size(chunk_size),
    m_array_size(array_size) {
    if (m_chunk_size.is_empty()) {
        throw std::runtime_error("chunk size cannot be empty");
    }

    size_t num_total_chunks = 1;

    for (size_t i = 0; i < N; i++) {
        m_num_chunks[i] = checked_cast<size_t>(div_ceil(array_size[i], m_chunk_size[i]));
        num_total_chunks *= m_num_chunks[i];
    }

    static constexpr size_t INVALID_INDEX = static_cast<size_t>(-1);
    std::vector<size_t> buffer_locs(num_total_chunks, INVALID_INDEX);

    for (const auto& chunk : chunks) {
        size_t buffer_index = 0;
        bool is_valid = true;
        point<N> expected_offset;
        dim<N> expected_size;

        for (size_t i = 0; i < N; i++) {
            auto k = div_floor(chunk.offset[i], m_chunk_size[i]);

            expected_offset[i] = k * m_chunk_size[i];
            expected_size[i] = std::min(m_chunk_size[i], array_size[i] - expected_offset[i]);

            buffer_index = buffer_index * m_num_chunks[i] + static_cast<size_t>(k);
        }

        if (chunk.offset != expected_offset || chunk.size != expected_size) {
            throw std::runtime_error(fmt::format(
                "invalid write access pattern, the region {} is not aligned to the chunk size of {}",
                rect<N>(expected_offset, expected_size),
                m_chunk_size));
        }

        if (buffer_locs[buffer_index] != INVALID_INDEX) {
            throw std::runtime_error(fmt::format(
                "invalid write access pattern, the region {} is written to by more one task",
                rect<N>(expected_offset, expected_size)));
        }

        buffer_locs[buffer_index] = buffer_index;
    }

    for (size_t index : buffer_locs) {
        if (index == INVALID_INDEX) {
            auto region = index2region(index, m_num_chunks, m_chunk_size, m_array_size);
            throw std::runtime_error(
                fmt::format("invalid write access pattern, no task writes to region {}", region));
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
        auto k = div_floor(region.offset[i], m_chunk_size[i]);
        auto w = region.offset[i] % m_chunk_size[i] + region.sizes[i];

        if (!in_range(k, m_num_chunks[i]) || w > m_chunk_size[i]) {
            throw std::out_of_range("invalid chunk");
        }

        buffer_index = buffer_index * m_num_chunks[i] + static_cast<size_t>(k);
        offset[i] = k * m_chunk_size[i];
        sizes[i] = m_chunk_size[i];
    }

    // TODO?
    MemoryId memory_id = MemoryId::host();

    return {m_buffers[buffer_index], memory_id, offset, sizes};
}

template<size_t N>
ArrayChunk<N> ArrayBackend<N>::chunk(size_t index) const {
    if (index >= m_buffers.size()) {
        throw std::runtime_error(fmt::format(
            "chunk {} is out of range, there are only {} chunks",
            index,
            m_buffers.size()));
    }

    // TODO?
    MemoryId memory_id = MemoryId::host();
    auto region = index2region(index, m_num_chunks, m_chunk_size, m_array_size);

    return {m_buffers[index], memory_id, region.offset, region.sizes};
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

    // Access each buffer once to check for errors.
    for (size_t i = 0; i < m_buffers.size(); i++) {
        auto memory_id = this->chunk(i).owner_id;
        m_worker->access_buffer(m_buffers[i], memory_id, AccessMode::Read);
    }
}

template<size_t N>
class CopyOutTask: public Task {
  public:
    CopyOutTask(void* data, size_t element_size, dim<N> array_size, rect<N> region) :
        m_dst_addr(data),
        m_element_size(element_size),
        m_array_size(array_size),
        m_region(region) {}

    void execute(ExecutionContext& proc, TaskContext context) override {
        KMM_ASSERT(context.accessors.size() == 1);
        const void* src_addr = context.accessors[0].address;
        size_t src_stride = 1;
        size_t dst_stride = 1;

        CopyDescription copy(m_element_size);

        for (size_t i = 0; i < N; i++) {
            copy.add_dimension(
                checked_cast<size_t>(m_region.size(i)),
                checked_cast<size_t>(0),
                checked_cast<size_t>(m_region.begin(i)),
                src_stride,
                dst_stride);

            src_stride *= checked_cast<size_t>(m_region.size(i));
            dst_stride *= checked_cast<size_t>(m_array_size.get(i));
        }

        execute_copy(src_addr, m_dst_addr, copy);
    }

  private:
    void* m_dst_addr;
    size_t m_element_size;
    dim<N> m_array_size;
    rect<N> m_region;
};

template<size_t N>
void ArrayBackend<N>::copy_bytes(void* dest_addr, size_t element_size) const {
    auto event_id = m_worker->with_task_graph([&](TaskGraph& graph) {
        EventList deps;

        for (size_t i = 0; i < m_buffers.size(); i++) {
            auto region = index2region(i, m_num_chunks, m_chunk_size, m_array_size);

            auto task =
                std::make_shared<CopyOutTask<N>>(dest_addr, element_size, m_array_size, region);
            auto buffer = BufferRequirement {
                .buffer_id = m_buffers[i],
                .memory_id = MemoryId::host(),
                .access_mode = AccessMode::Read,
            };

            auto event_id = graph.insert_task(ProcessorId::host(), std::move(task), {buffer});

            deps.push_back(event_id);
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