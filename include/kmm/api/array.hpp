#pragma once

#include <memory>
#include <vector>

#include "spdlog/spdlog.h"

#include "kmm/api/access.hpp"
#include "kmm/api/argument.hpp"
#include "kmm/api/array_argument.hpp"
#include "kmm/api/array_backend.hpp"
#include "kmm/api/array_builder.hpp"

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

template<typename T>
using Scalar = Array<T, 0>;

template<typename T, size_t N, typename A>
struct ArgumentHandler<Read<Array<T, N>, A>> {
    using type =
        ArrayArgument<const T, views::layouts::right_to_left<views::domains::subbounds<N>>>;

    static_assert(
        is_dimensionality_accepted_by_mapper<A, N>,
        "mapper of 'read' must return N-dimensional region"
    );

    ArgumentHandler(Read<Array<T, N>, A> arg) :
        m_backend(arg.argument.inner().shared_from_this()),
        m_access_map(arg.access_map) {}

    type process_chunk(TaskChunk chunk, TaskBuilder& builder) {
        auto access_region = m_access_map(chunk, m_backend->array_size());
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
    A m_access_map;
};

template<typename T, size_t N, typename A>
struct ArgumentHandler<Write<Array<T, N>, A>> {
    using type = ArrayArgument<T, views::layouts::right_to_left<views::domains::subbounds<N>>>;

    static_assert(
        is_dimensionality_accepted_by_mapper<A, N>,
        "mapper of 'write' must return N-dimensional region"
    );

    ArgumentHandler(Write<Array<T, N>, A> arg) :
        m_array(arg.argument),
        m_access_map(arg.access_map),
        m_builder(arg.argument.shape(), BufferLayout::for_type<T>()) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    type process_chunk(TaskChunk chunk, TaskBuilder& builder) {
        auto access_region = m_access_map(chunk, m_builder.m_sizes);
        size_t buffer_index = m_builder.add_chunk(builder, access_region);
        views::domains::subbounds<N> domain = {access_region.offset, access_region.sizes};
        return {buffer_index, domain};
    }

    void finalize(const TaskResult& result) {
        m_array = Array<T, N>(m_builder.build(result.worker, result.graph));
    }

  private:
    Array<T, N>& m_array;
    A m_access_map;
    ArrayBuilder<N> m_builder;
};

template<typename T, size_t N>
struct ArgumentHandler<Read<Array<T, N>>> {
    using type = ArrayArgument<const T, views::layouts::right_to_left<views::domains::bounds<N>>>;

    ArgumentHandler(Read<Array<T, N>> arg) : m_backend(arg.argument.inner().shared_from_this()) {}

    type process_chunk(TaskChunk chunk, TaskBuilder& builder) {
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
struct ArgumentHandler<Write<Array<T, N>>> {
    using type = ArrayArgument<T, views::layouts::right_to_left<views::domains::bounds<N>>>;

    ArgumentHandler(Write<Array<T, N>> arg) :
        m_array(arg.argument),
        m_builder(arg.argument.shape(), BufferLayout::for_type<T>()) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    type process_chunk(TaskChunk chunk, TaskBuilder& builder) {
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
    ArrayBuilder<N> m_builder;
};

template<typename T, size_t N>
struct ArgumentHandler<Array<T, N>>: public ArgumentHandler<Read<Array<T, N>>> {
    ArgumentHandler(Array<T, N> arg) : ArgumentHandler<Read<Array<T, N>>>(read(arg)) {}
};

template<typename T, size_t N>
struct ArgumentHandler<Reduce<Array<T, N>>> {
    using type = ArrayArgument<T, views::layouts::right_to_left<views::domains::bounds<N>>>;

    ArgumentHandler(Reduce<Array<T, N>> arg) :
        m_array(arg.argument),
        m_builder(arg.argument.shape(), DataType::of<T>(), arg.op) {}

    type process_chunk(TaskChunk chunk, TaskBuilder& builder) {
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

template<typename T, size_t N, typename P>
struct ArgumentHandler<Reduce<Array<T, N>, All, P>> {
    static constexpr size_t K = mapper_dimensionality<P>;
    static_assert(
        is_dimensionality_accepted_by_mapper<P, K>,
        "private mapper of 'reduce' must return N-dimensional region"
    );

    using type = ArrayArgument<T, views::layouts::right_to_left<views::domains::subbounds<K + N>>>;

    ArgumentHandler(Reduce<Array<T, N>, All, P> arg) :
        m_array(arg.argument),
        m_builder(arg.argument.shape(), DataType::of<T>(), arg.op),
        m_private_map(arg.private_map) {}

    type process_chunk(TaskChunk chunk, TaskBuilder& builder) {
        auto access_region = Rect<N>(m_builder.m_sizes);
        auto private_region = m_private_map(chunk);
        size_t buffer_index = m_builder.add_chunk(builder, access_region, private_region.size());

        views::domains::subbounds<K + N> domain = {
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
    P m_private_map;
};

template<typename T, size_t N, typename A, typename P>
struct ArgumentHandler<Reduce<Array<T, N>, A, P>> {
    static constexpr size_t K = mapper_dimensionality<P>;
    using type = ArrayArgument<T, views::layouts::right_to_left<views::domains::subbounds<K + N>>>;

    static_assert(
        is_dimensionality_accepted_by_mapper<A, N>,
        "mapper of 'reduce' must return N-dimensional region"
    );

    static_assert(
        is_dimensionality_accepted_by_mapper<P, K>,
        "private mapper of 'reduce' must return N-dimensional region"
    );

    ArgumentHandler(Reduce<Array<T, N>, A, P> arg) :
        m_array(arg.argument),
        m_builder(arg.argument.shape(), DataType::of<T>(), arg.op),
        m_access_map(arg.access_map),
        m_private_map(arg.private_map) {}

    type process_chunk(TaskChunk chunk, TaskBuilder& builder) {
        auto private_region = m_private_map(chunk);
        auto access_region = m_access_map(chunk, m_builder.m_sizes);
        size_t buffer_index = m_builder.add_chunk(builder, access_region, private_region.size());

        views::domains::subbounds<K + N> domain = {
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
    A m_access_map;
    P m_private_map;
};

}  // namespace kmm