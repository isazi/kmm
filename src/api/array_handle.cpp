#include "spdlog/spdlog.h"

#include "kmm/api/array.hpp"
#include "kmm/memops/gpu_copy.hpp"
#include "kmm/memops/host_copy.hpp"
#include "kmm/utils/integer_fun.hpp"
#include "kmm/worker/worker.hpp"

namespace kmm {

template<size_t N>
Range<N> index2region(
    size_t index,
    std::array<size_t, N> num_chunks,
    Size<N> chunk_size,
    Size<N> array_size
) {
    Index<N> offset;
    Size<N> sizes;

    for (size_t j = 0; compare_less(j, N); j++) {
        size_t i = N - 1 - j;
        auto k = index % num_chunks[i];
        index /= num_chunks[i];

        offset[i] = int64_t(k) * chunk_size[i];
        sizes[i] = std::min(chunk_size[i], array_size[i] - offset[i]);
    }

    return {offset, sizes};
}

template<size_t N>
DataDistribution<N>::DataDistribution(Size<N> array_size, std::vector<DataChunk<N>> chunks) :
    m_array_size(array_size) {
    for (const auto& chunk : chunks) {
        if (chunk.offset == Index<N>::zero()) {
            m_chunk_size = chunk.size;
        }
    }

    if (m_chunk_size.is_empty()) {
        throw std::runtime_error("chunk size cannot be empty");
    }

    for (size_t i = 0; compare_less(i, N); i++) {
        m_chunks_count[i] = checked_cast<size_t>(div_ceil(array_size[i], m_chunk_size[i]));
    }

    size_t num_total_chunks = checked_product(m_chunks_count.begin(), m_chunks_count.end());

    static constexpr size_t INVALID_INDEX = static_cast<size_t>(-1);
    std::vector<size_t> buffer_locs(num_total_chunks, INVALID_INDEX);

    for (const auto& chunk : chunks) {
        size_t buffer_index = 0;
        Index<N> expected_offset;
        Size<N> expected_size;

        for (size_t i = 0; compare_less(i, N); i++) {
            auto k = div_floor(chunk.offset[i], m_chunk_size[i]);

            expected_offset[i] = k * m_chunk_size[i];
            expected_size[i] = std::min(m_chunk_size[i], array_size[i] - expected_offset[i]);

            buffer_index = buffer_index * m_chunks_count[i] + static_cast<size_t>(k);
        }

        if (chunk.offset != expected_offset || chunk.size != expected_size) {
            throw std::runtime_error(fmt::format(
                "invalid write access pattern, the region {} is not aligned to the chunk size of {}",
                Range<N>(chunk.offset, chunk.size),
                m_chunk_size
            ));
        }

        if (buffer_locs[buffer_index] != INVALID_INDEX) {
            throw std::runtime_error(fmt::format(
                "invalid write access pattern, the region {} is written to by more one task",
                Range<N>(expected_offset, expected_size)
            ));
        }

        buffer_locs[buffer_index] = buffer_index;
    }

    for (size_t index : buffer_locs) {
        if (index == INVALID_INDEX) {
            auto region = index2region(index, m_chunks_count, m_chunk_size, m_array_size);
            throw std::runtime_error(
                fmt::format("invalid write access pattern, no task writes to region {}", region)
            );
        }
    }

    for (size_t index : buffer_locs) {
        m_chunks.push_back(chunks[index]);
    }
}

template<size_t N>
ArrayHandle<N>::~ArrayHandle() {
    m_worker->with_task_graph([&](auto& builder) {
        for (const auto& chunk : this->m_chunks) {
            builder.delete_buffer(chunk.buffer_id);
        }
    });
}

template<size_t N>
DataChunk<N> DataDistribution<N>::find_chunk(Range<N> region) const {
    size_t index = 0;
    Index<N> offset;
    Size<N> sizes;

    for (size_t i = 0; compare_less(i, N); i++) {
        auto k = div_floor(region.offset[i], m_chunk_size[i]);
        auto w = region.offset[i] % m_chunk_size[i] + region.sizes[i];

        if (!in_range(k, m_chunks_count[i])) {
            throw std::out_of_range(fmt::format(
                "invalid read pattern, the region {} exceeds the array dimensions {}",
                region,
                m_array_size
            ));
        }

        if (w > m_chunk_size[i]) {
            throw std::out_of_range(fmt::format(
                "invalid read pattern, the region {} does not align to the chunk size of {}",
                region,
                m_chunk_size
            ));
        }

        index = index * m_chunks_count[i] + static_cast<size_t>(k);
        offset[i] = k * m_chunk_size[i];
        sizes[i] = m_chunk_size[i];
    }

    // TODO?
    MemoryId memory_id = MemoryId::host();

    return {m_chunks[index].buffer_id, memory_id, offset, sizes};
}

template<size_t N>
DataChunk<N> DataDistribution<N>::chunk(size_t index) const {
    if (index >= m_chunks.size()) {
        throw std::runtime_error(fmt::format(
            "chunk {} is out of range, there are only {} chunks",
            index,
            m_chunks.size()
        ));
    }

    return m_chunks[index];
}

template<size_t N>
void ArrayHandle<N>::synchronize() const {
    auto event_id = m_worker->with_task_graph([&](TaskGraph& graph) {
        auto deps = EventList {};

        for (const auto& chunk : this->m_chunks) {
            deps.insert_all(graph.extract_buffer_dependencies(chunk.buffer_id));
        }

        return graph.join_events(deps);
    });

    m_worker->query_event(event_id, std::chrono::system_clock::time_point::max());

    // Access each buffer once to check for errors.
    for (size_t i = 0; i < this->m_chunks.size(); i++) {
        //        auto memory_id = this->chunk(i).owner_id;
        //        m_worker->access_buffer(m_buffers[i], memory_id, AccessMode::Read);
        //        KMM_TODO();
    }
}

template<size_t N>
class CopyOutTask: public Task {
  public:
    CopyOutTask(void* data, size_t element_size, Size<N> array_size, Range<N> region) :
        m_dst_addr(data) {
        size_t src_stride = 1;
        size_t dst_stride = 1;

        CopyDef copy(element_size);

        for (size_t j = 0; compare_less(j, N); j++) {
            size_t i = N - j - 1;

            copy.add_dimension(
                checked_cast<size_t>(region.size(i)),
                checked_cast<size_t>(0),
                checked_cast<size_t>(region.begin(i)),
                src_stride,
                dst_stride
            );

            src_stride *= checked_cast<size_t>(region.size(i));
            dst_stride *= checked_cast<size_t>(array_size.get(i));
        }
    }

    void execute(ExecutionContext& proc, TaskContext context) override {
        KMM_ASSERT(context.accessors.size() == 1);
        const void* src_addr = context.accessors[0].address;
        execute_copy(src_addr, m_dst_addr, m_copy);
    }

  private:
    void* m_dst_addr;
    CopyDef m_copy;
};

template<size_t N>
void ArrayHandle<N>::copy_bytes(void* dest_addr, size_t element_size) const {
    auto dest_mem = MemoryId::host();

    if (auto ordinal = get_gpu_device_by_address(dest_addr)) {
        dest_mem = m_worker->system_info().device_by_ordinal(*ordinal).memory_id();
    }

    auto event_id = m_worker->with_task_graph([&](TaskGraph& graph) {
        EventList deps;

        for (size_t i = 0; i < this->m_chunks.size(); i++) {
            auto& chunk = this->m_chunks[i];
            auto region = Range<N> {chunk.offset, chunk.size};

            auto task = std::make_shared<CopyOutTask<N>>(
                dest_addr,
                element_size,
                this->m_array_size,
                region
            );

            auto buffer = BufferRequirement {
                .buffer_id = chunk.buffer_id,
                .memory_id = MemoryId::host(),
                .access_mode = AccessMode::Read,
            };

            auto event_id = graph.insert_task(ProcessorId::host(), std::move(task), {buffer});
            deps.push_back(event_id);
        }

        return graph.join_events(std::move(deps));
    });

    m_worker->query_event(event_id, std::chrono::system_clock::time_point::max());
}

template<size_t N>
ArrayHandle<N>::ArrayHandle(Worker& worker, DataDistribution<N> distribution) :
    DataDistribution<N>(std::move(distribution)),
    m_worker(worker.shared_from_this()) {}

#define INSTANTIATE_ARRAY_IMPL(NAME)     \
    template class NAME<0>; /* NOLINT */ \
    template class NAME<1>; /* NOLINT */ \
    template class NAME<2>; /* NOLINT */ \
    template class NAME<3>; /* NOLINT */ \
    template class NAME<4>; /* NOLINT */ \
    template class NAME<5>; /* NOLINT */ \
    template class NAME<6>; /* NOLINT */

INSTANTIATE_ARRAY_IMPL(DataDistribution)
INSTANTIATE_ARRAY_IMPL(ArrayHandle)

}  // namespace kmm