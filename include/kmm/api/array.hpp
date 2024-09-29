#pragma once

#include <memory>
#include <vector>

#include "spdlog/spdlog.h"

#include "kmm/api/access.hpp"
#include "kmm/api/array_backend.hpp"
#include "kmm/api/packed_array.hpp"
#include "kmm/api/task_data.hpp"

namespace kmm {

class ArrayBase {
  public:
    virtual ~ArrayBase() = default;
    virtual const std::type_info& type_info() const = 0;
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

    const std::type_info& type_info() const final {
        return typeid(T);
    }

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
        m_mapping(arg.slice_mapping) {}

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
        m_mapping(arg.slice_mapping),
        m_builder(arg.argument.shape(), BufferLayout::for_type<T>()) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    type process_chunk(Chunk chunk, TaskBuilder& builder) {
        auto access_region = m_mapping(chunk, m_builder.m_sizes);
        size_t buffer_index = m_builder.add_chunk(builder, access_region);
        views::domains::subbounds<N> domain = {access_region.offset, access_region.sizes};
        return {buffer_index, domain};
    }

    void finalize(const TaskResult& result) {
        m_array = Array<T, N>(m_builder.build(result.worker));
    }

  private:
    Array<T, N>& m_array;
    SliceMapping<N> m_mapping;
    ArrayBuilder<N> m_builder;
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
        m_builder(arg.argument.shape(), BufferLayout::for_type<T>()) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    type process_chunk(Chunk chunk, TaskBuilder& builder) {
        auto access_region = m_builder.m_sizes;
        size_t buffer_index = m_builder.add_chunk(builder, access_region);
        views::domains::bounds<N> domain = {access_region};

        return {buffer_index, domain};
    }

    void finalize(const TaskResult& result) {
        m_array = Array<T, N>(m_builder.build(result.worker));
    }

  private:
    Array<T, N>& m_array;
    ArrayBuilder<N> m_builder;
};

template<typename T, size_t N>
struct TaskDataProcessor<Array<T, N>>: public TaskDataProcessor<Read<Array<T, N>>> {
    TaskDataProcessor(Array<T, N> arg) : TaskDataProcessor<Read<Array<T, N>>>(read(arg)) {}
};

template<typename T, size_t N>
struct TaskDataProcessor<Reduce<Array<T, N>>> {
    using type = PackedArray<T, views::layouts::right_to_left<views::domains::bounds<N>>>;

    TaskDataProcessor(Reduce<Array<T, N>, FullMapping> arg) :
        m_array(arg.argument),
        m_builder(arg.argument.shape(), DataType::of<T>(), arg.op) {}

    type process_chunk(Chunk chunk, TaskBuilder& builder) {
        auto access_region = m_builder.m_sizes;
        size_t buffer_index = m_builder.add_chunk(builder, access_region);
        views::domains::bounds<N> domain = {access_region};
        return {buffer_index, domain};
    }

    void finalize(const TaskResult& result) {
        m_array = Array<T, N>(m_builder.build(result.worker, result.graph));
    }

  private:
    Array<T, N>& m_array;
    ArrayReductionBuilder<N> m_builder;
};

template<typename T, size_t P, size_t N>
struct TaskDataProcessor<Reduce<Array<T, N>, FullMapping, SliceMapping<P>>> {
    using type = PackedArray<T, views::layouts::right_to_left<views::domains::subbounds<P + N>>>;

    TaskDataProcessor(Reduce<Array<T, N>, FullMapping, SliceMapping<P>> arg) :
        m_array(arg.argument),
        m_builder(arg.argument.shape(), DataType::of<T>(), arg.op),
        m_private_mapping(arg.private_mapping) {}

    type process_chunk(Chunk chunk, TaskBuilder& builder) {
        auto private_region = m_private_mapping(chunk);
        auto access_region = Rect<N>(m_builder.m_sizes);
        size_t buffer_index = m_builder.add_chunk(builder, access_region, private_region.size());

        views::domains::subbounds<P + N> domain = {
            private_region.offset.concat(access_region.offset),
            private_region.sizes.concat(access_region.sizes),
        };

        return {buffer_index, domain};
    }

    void finalize(const TaskResult& result) {
        m_array = Array<T, N>(m_builder.build(result.worker, result.graph));
    }

  private:
    Array<T, N>& m_array;
    ArrayReductionBuilder<N> m_builder;
    SliceMapping<P> m_private_mapping;
};

template<typename T, size_t P, size_t N>
struct TaskDataProcessor<Reduce<Array<T, N>, SliceMapping<N>, SliceMapping<P>>> {
    using type = PackedArray<T, views::layouts::right_to_left<views::domains::subbounds<P + N>>>;

    TaskDataProcessor(Reduce<Array<T, N>, SliceMapping<N>, SliceMapping<P>> arg) :
        m_array(arg.argument),
        m_builder(arg.argument.shape(), DataType::of<T>(), arg.op),
        m_mapping(arg.slice_mapping),
        m_private_mapping(arg.private_mapping) {}

    type process_chunk(Chunk chunk, TaskBuilder& builder) {
        auto private_region = m_private_mapping(chunk);
        auto access_region = m_mapping(chunk, m_builder.m_sizes);
        size_t buffer_index = m_builder.add_chunk(builder, access_region, private_region.size());

        views::domains::subbounds<P + N> domain = {
            private_region.offset.concat(access_region.offset),
            private_region.sizes.concat(access_region.sizes),
        };

        return {buffer_index, domain};
    }

    void finalize(const TaskResult& result) {
        m_array = Array<T, N>(m_builder.build(result.worker, result.graph));
    }

  private:
    Array<T, N>& m_array;
    ArrayReductionBuilder<N> m_builder;
    SliceMapping<N> m_mapping;
    SliceMapping<P> m_private_mapping;
};

}  // namespace kmm