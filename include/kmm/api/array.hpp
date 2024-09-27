#pragma once

#include <memory>
#include <vector>

#include "spdlog/spdlog.h"

#include "kmm/api/access.hpp"
#include "kmm/api/packed_array.hpp"
#include "kmm/api/task_data.hpp"
#include "kmm/core/geometry.hpp"
#include "kmm/core/identifiers.hpp"
#include "kmm/internals/scheduler.hpp"
#include "kmm/internals/task_graph.hpp"

namespace kmm {

class Worker;

template<size_t N>
struct ArrayChunk {
    BufferId buffer_id;
    MemoryId owner_id;
    Point<N> offset;
    Dim<N> size;
};

template<size_t N>
class ArrayBackend: public std::enable_shared_from_this<ArrayBackend<N>> {
  public:
    ArrayBackend(
        std::shared_ptr<Worker> worker,
        Dim<N> array_size,
        std::vector<ArrayChunk<N>> chunks);
    ~ArrayBackend();

    ArrayChunk<N> find_chunk(Rect<N> region) const;
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

    Dim<N> chunk_size() const {
        return m_chunk_size;
    }

    Dim<N> array_size() const {
        return m_array_size;
    }

  private:
    std::shared_ptr<Worker> m_worker;
    std::vector<BufferId> m_buffers;
    Dim<N> m_array_size;
    Dim<N> m_chunk_size;
    std::array<size_t, N> m_num_chunks;
};

class ArrayBase {
  public:
    virtual ~ArrayBase() = default;
    virtual size_t rank() const = 0;
    virtual int64_t size(size_t axis) const = 0;
};

template<typename T, size_t N = 1>
class Array: ArrayBase {
  public:
    Array(Dim<N> shape = {}) : m_shape(shape) {}

    explicit Array(std::shared_ptr<ArrayBackend<N>> b) :
        m_backend(b),
        m_shape(m_backend->array_size()) {}

    size_t rank() const final {
        return N;
    }

    Dim<N> shape() const {
        return m_shape;
    }

    int64_t size(size_t axis) const final {
        return m_shape.get(axis);
    }

    int64_t size() const {
        return m_shape.volume();
    }

    bool is_empty() const {
        return m_shape.is_empty();
    }

    bool is_valid() const {
        return m_backend != nullptr;
    }

    const ArrayBackend<N>& inner() const {
        KMM_ASSERT(m_backend != nullptr);
        return *m_backend;
    }

    Dim<N> chunk_size() const {
        return inner().chunk_size();
    }

    int64_t chunk_size(size_t axis) const {
        return inner().chunk_size().get(axis);
    }

    const Worker& worker() const {
        return inner().worker();
    }

    void synchronize() const {
        if (m_backend) {
            m_backend->synchronize();
        }
    }

    void clear() {
        m_backend = nullptr;
    }

    void copy_to(T* output) const {
        inner().copy_bytes(output, sizeof(T));
    }

  private:
    std::shared_ptr<ArrayBackend<N>> m_backend;
    Dim<N> m_shape;
};

template<typename T, size_t N>
struct TaskDataProcessor<Read<Array<T, N>, SliceMapping<N>>> {
    using type = PackedArray<const T, views::layouts::right_to_left<views::domains::subbounds<N>>>;

    TaskDataProcessor(Read<Array<T, N>, SliceMapping<N>> arg) :
        m_backend(arg.argument.inner().shared_from_this()),
        m_mapping(arg.index_mapping) {}

    type process_chunk(Chunk chunk, TaskBuilder& builder) {
        auto access_region = m_mapping(chunk, m_backend->array_size());
        auto data_chunk = m_backend->find_chunk(access_region);

        size_t buffer_index = builder.buffers.size();
        builder.buffers.emplace_back(BufferRequirement {
            .buffer_id = data_chunk.buffer_id,
            .memory_id = builder.memory_id,
            .access_mode = AccessMode::Read});

        auto domain = views::domains::subbounds<N> {data_chunk.offset, data_chunk.size};
        return {buffer_index, domain};
    }

    void finalize(const TaskResult& result) {
        // Nothing to do
    }

  private:
    std::shared_ptr<const ArrayBackend<N>> m_backend;
    SliceMapping<N> m_mapping;
};

template<typename T, size_t N>
struct TaskDataProcessor<Write<Array<T, N>, SliceMapping<N>>> {
    using type = PackedArray<T, views::layouts::right_to_left<views::domains::subbounds<N>>>;

    TaskDataProcessor(Write<Array<T, N>, SliceMapping<N>> arg) :
        m_array(arg.argument),
        m_mapping(arg.index_mapping),
        m_sizes(arg.argument.shape()) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    type process_chunk(Chunk chunk, TaskBuilder& builder) {
        auto access_region = m_mapping(chunk, m_sizes);
        auto num_elements = access_region.size();
        auto buffer_id = builder.graph.create_buffer(BufferLayout::for_type<T>(num_elements));

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

        views::domains::subbounds<N> domain = {access_region.offset, access_region.sizes};
        return {buffer_index, domain};
    }

    void finalize(const TaskResult& result) {
        std::shared_ptr<Worker> worker = result.worker.shared_from_this();
        m_array =
            Array<T, N>(std::make_shared<ArrayBackend<N>>(worker, m_sizes, std::move(m_chunks)));
    }

  private:
    Array<T, N>& m_array;
    Dim<N> m_sizes;
    std::vector<ArrayChunk<N>> m_chunks;
    SliceMapping<N> m_mapping;
};

template<typename T, size_t N>
struct TaskDataProcessor<Read<Array<T, N>>> {
    using type = PackedArray<const T, views::layouts::right_to_left<views::domains::bounds<N>>>;

    TaskDataProcessor(Read<Array<T, N>> arg) : m_backend(arg.argument.inner().shared_from_this()) {}

    type process_chunk(Chunk chunk, TaskBuilder& builder) {
        auto data_chunk = m_backend->find_chunk(m_backend->array_size());

        size_t buffer_index = builder.buffers.size();
        builder.buffers.emplace_back(BufferRequirement {
            .buffer_id = data_chunk.buffer_id,
            .memory_id = builder.memory_id,
            .access_mode = AccessMode::Read});

        auto domain = views::domains::bounds<N> {data_chunk.size};
        return {buffer_index, domain};
    }

    void finalize(const TaskResult& result) {
        // Nothing to do
    }

  private:
    std::shared_ptr<const ArrayBackend<N>> m_backend;
};

template<typename T, size_t N>
struct TaskDataProcessor<Write<Array<T, N>>> {
    using type = PackedArray<T, views::layouts::right_to_left<views::domains::bounds<N>>>;

    TaskDataProcessor(Write<Array<T, N>> arg) :
        m_array(arg.argument),
        m_sizes(arg.argument.shape()) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    type process_chunk(Chunk chunk, TaskBuilder& builder) {
        auto num_elements = m_sizes.volume();
        auto buffer_id = builder.graph.create_buffer(BufferLayout::for_type<T>(num_elements));

        m_chunks.push_back(ArrayChunk<N> {
            .buffer_id = buffer_id,
            .owner_id = chunk.owner_id.as_memory(),
            .offset = Point<N>::zero(),
            .size = m_sizes});

        auto domain = views::domains::bounds<N> {m_sizes};

        size_t buffer_index = builder.buffers.size();
        builder.buffers.emplace_back(BufferRequirement {
            .buffer_id = buffer_id,
            .memory_id = builder.memory_id,
            .access_mode = AccessMode::Exclusive});

        return {buffer_index, domain};
    }

    void finalize(const TaskResult& result) {
        std::shared_ptr<Worker> worker = result.worker.shared_from_this();
        m_array =
            Array<T, N>(std::make_shared<ArrayBackend<N>>(worker, m_sizes, std::move(m_chunks)));
    }

  private:
    Array<T, N>& m_array;
    Dim<N> m_sizes;
    std::vector<ArrayChunk<N>> m_chunks;
};

template<typename T, size_t N>
struct TaskDataProcessor<Array<T, N>>: public TaskDataProcessor<Read<Array<T, N>>> {
    TaskDataProcessor(Array<T, N> arg) : TaskDataProcessor<Read<Array<T, N>>>(read(arg)) {}
};

template<typename T, size_t N>
struct TaskDataProcessor<Reduce<Array<T, N>>> {
    using type = PackedArray<T, views::layouts::right_to_left<views::domains::bounds<N>>>;

    TaskDataProcessor(Reduce<Array<T, N>> arg) :
        m_array(arg.argument),
        m_sizes(arg.argument.shape()),
        m_reduction(arg.op) {}

    type process_chunk(Chunk chunk, TaskBuilder& builder) {
        auto num_elements = m_sizes.volume();

        auto layout = BufferLayout::for_type<T>(num_elements);
        layout.fill_pattern = reduction_identity_value(DataType::of<T>(), m_reduction);

        auto buffer_id = builder.graph.create_buffer(std::move(layout));
        auto memory_id = builder.memory_id;

        m_partial_inputs.push_back(ReductionInput {
            .buffer_id = buffer_id,
            .memory_id = memory_id,
            .dependencies = {},
            .num_inputs_per_output = 1});

        size_t buffer_index = builder.buffers.size();
        builder.buffers.emplace_back(BufferRequirement {
            .buffer_id = buffer_id,
            .memory_id = memory_id,
            .access_mode = AccessMode::Exclusive});

        views::domains::bounds<N> domain = {m_sizes};
        return {buffer_index, domain};
    }

    void finalize(const TaskResult& result) {
        auto num_elements = m_sizes.volume();
        auto buffer_id = result.graph.create_buffer(BufferLayout::for_type<T>(num_elements));

        MemoryId memory_id = m_partial_inputs[0].memory_id;
        auto event_id = result.graph.insert_reduction(
            m_reduction,
            buffer_id,
            memory_id,
            DataType::of<T>(),
            num_elements,
            m_partial_inputs);

        std::vector<ArrayChunk<N>> chunks = {
            {.buffer_id = buffer_id,
             .owner_id = memory_id,
             .offset = Point<N>::zero(),
             .size = m_sizes}};

        for (const auto& input : m_partial_inputs) {
            result.graph.delete_buffer(input.buffer_id, {event_id});
        }

        std::shared_ptr<Worker> worker = result.worker.shared_from_this();
        m_array =
            Array<T, N>(std::make_shared<ArrayBackend<N>>(worker, m_sizes, std::move(chunks)));
    }

  private:
    Array<T, N>& m_array;
    Dim<N> m_sizes;
    ReductionOp m_reduction;
    std::vector<ReductionInput> m_partial_inputs;
};

template<typename T, size_t N>
struct TaskDataProcessor<Reduce<Array<T, N>, SliceMapping<N>>> {
    using type = PackedArray<T, views::layouts::right_to_left<views::domains::subbounds<N>>>;

    TaskDataProcessor(Reduce<Array<T, N>, SliceMapping<N>> arg) :
        m_array(arg.argument),
        m_sizes(arg.argument.shape()),
        m_reduction(arg.op),
        m_mapping(arg.index_mapping) {}

    type process_chunk(Chunk chunk, TaskBuilder& builder) {
        auto access_region = m_mapping(chunk, m_sizes);
        auto num_elements = access_region.size();

        auto layout = BufferLayout::for_type<T>(num_elements);
        layout.fill_pattern = reduction_identity_value(DataType::of<T>(), m_reduction);

        auto buffer_id = builder.graph.create_buffer(std::move(layout));
        auto memory_id = builder.memory_id;

        m_partial_inputs[access_region].push_back(ReductionInput {
            .buffer_id = buffer_id,
            .memory_id = memory_id,
            .dependencies = {},
            .num_inputs_per_output = 1});

        size_t buffer_index = builder.buffers.size();
        builder.buffers.emplace_back(BufferRequirement {
            .buffer_id = buffer_id,
            .memory_id = memory_id,
            .access_mode = AccessMode::Exclusive});

        views::domains::subbounds<N> domain = {access_region.offset, access_region.sizes};
        return {buffer_index, domain};
    }

    void finalize(const TaskResult& result) {
        std::vector<ArrayChunk<N>> chunks;

        for (auto& p : m_partial_inputs) {
            auto access_region = p.first;
            auto& inputs = p.second;

            MemoryId memory_id = inputs[0].memory_id;
            auto num_elements = access_region.size();

            auto buffer_id = result.graph.create_buffer(BufferLayout::for_type<T>(num_elements));

            auto event_id = result.graph.insert_reduction(
                m_reduction,
                buffer_id,
                memory_id,
                DataType::of<T>(),
                num_elements,
                inputs);

            chunks.push_back(ArrayChunk<N> {
                .buffer_id = buffer_id,
                .owner_id = memory_id,
                .offset = access_region.offset,
                .size = access_region.sizes});

            for (const auto& input : inputs) {
                result.graph.delete_buffer(input.buffer_id, {event_id});
            }
        }

        std::shared_ptr<Worker> worker = result.worker.shared_from_this();
        m_array =
            Array<T, N>(std::make_shared<ArrayBackend<N>>(worker, m_sizes, std::move(chunks)));
    }

  private:
    Array<T, N>& m_array;
    Dim<N> m_sizes;
    ReductionOp m_reduction;
    SliceMapping<N> m_mapping;
    std::unordered_map<Rect<N>, std::vector<ReductionInput>> m_partial_inputs;
};

}  // namespace kmm