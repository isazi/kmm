#pragma once

#include <memory>
#include <stdexcept>
#include <typeinfo>
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
    virtual void copy_bytes_to(void* output, size_t num_bytes) const = 0;
};

template<typename T, size_t N = 1>
class Array: public ArrayBase {
  public:
    Array(Size<N> shape = {}) : m_shape(shape) {}

    explicit Array(std::shared_ptr<ArrayBackend<N>> b) :
        m_backend(b),
        m_shape(m_backend->array_size()) {}

    const std::type_info& type_info() const final {
        return typeid(T);
    }

    size_t rank() const final {
        return N;
    }

    Size<N> sizes() const {
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

    Size<N> chunk_size() const {
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

    void reset() {
        m_backend = nullptr;
    }

    void copy_bytes_to(void* output, size_t num_bytes) const {
        KMM_ASSERT(num_bytes % sizeof(T) == 0 && compare_equal(num_bytes / sizeof(T), size()));
        inner().copy_bytes(output, sizeof(T));
    }

    void copy_to(T* output) const {
        inner().copy_bytes(output, sizeof(T));
    }

    template<typename I>
    void copy_to(T* output, I num_elements) const {
        KMM_ASSERT(compare_equal(num_elements, size()));
        inner().copy_bytes(output, sizeof(T));
    }

    void copy_to(std::vector<T>& output) const {
        output.resize(checked_cast<size_t>(size()));
        inner().copy_bytes(output, sizeof(T));
    }

    std::vector<T> copy() const {
        std::vector<T> output;
        copy_to(output);
        return output;
    }

  private:
    std::shared_ptr<ArrayBackend<N>> m_backend;
    Index<N> m_offset;  // Unused for now
    Size<N> m_shape;
};

template<typename T>
using Scalar = Array<T, 0>;

template<typename T, size_t N>
struct ArgumentHandler<Read<Array<T, N>>> {
    using type = ArrayArgument<const T, views::domain_bounds<N>>;

    ArgumentHandler(Read<Array<T, N>> arg) :
        m_backend(arg.argument.inner().shared_from_this()),
        m_chunk(m_backend->find_chunk(m_backend->array_size())) {}

    void initialize(const TaskInit& init) {}

    type process_chunk(TaskBuilder& builder) {
        size_t buffer_index = builder.buffers.size();
        builder.buffers.emplace_back(BufferRequirement {
            .buffer_id = m_chunk.buffer_id,
            .memory_id = builder.memory_id,
            .access_mode = AccessMode::Read});

        auto domain = views::domain_bounds<N> {m_chunk.size};
        return {buffer_index, domain};
    }

    void finalize(const TaskResult& result) {}

  private:
    std::shared_ptr<const ArrayBackend<N>> m_backend;
    ArrayChunk<N> m_chunk;
};

template<typename T, size_t N>
struct ArgumentHandler<Write<Array<T, N>>> {
    using type = ArrayArgument<T, views::domain_bounds<N>>;

    ArgumentHandler(Write<Array<T, N>> arg) :
        m_array(arg.argument),
        m_builder(arg.argument.sizes(), BufferLayout::for_type<T>()) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    void initialize(const TaskInit& init) {}

    type process_chunk(TaskBuilder& builder) {
        auto access_region = m_builder.sizes();
        size_t buffer_index = m_builder.add_chunk(builder, access_region);
        views::domain_bounds<N> domain = {access_region};

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

template<typename T, size_t N, typename A>
struct ArgumentHandler<Read<Array<T, N>, A>> {
    using type = ArrayArgument<const T, views::domain_subbounds<N>>;

    static_assert(
        is_dimensionality_accepted_by_mapper<A, N>,
        "mapper of 'read' must return N-dimensional region"
    );

    ArgumentHandler(Read<Array<T, N>, A> arg) :
        m_backend(arg.argument.inner().shared_from_this()),
        m_access_mapper(arg.access_mapper) {}

    void initialize(const TaskInit& init) {}

    type process_chunk(TaskBuilder& builder) {
        auto access_region = m_access_mapper(builder.chunk, m_backend->array_size());
        auto data_chunk = m_backend->find_chunk(access_region);

        size_t buffer_index = builder.buffers.size();
        builder.buffers.emplace_back(BufferRequirement {
            .buffer_id = data_chunk.buffer_id,
            .memory_id = builder.memory_id,
            .access_mode = AccessMode::Read});

        auto domain = views::domain_subbounds<N> {data_chunk.offset, data_chunk.size};
        return {buffer_index, domain};
    }

    void finalize(const TaskResult& result) {}

  private:
    std::shared_ptr<const ArrayBackend<N>> m_backend;
    A m_access_mapper;
};

template<typename T, size_t N, typename A>
struct ArgumentHandler<Write<Array<T, N>, A>> {
    using type = ArrayArgument<T, views::domain_subbounds<N>>;

    static_assert(
        is_dimensionality_accepted_by_mapper<A, N>,
        "mapper of 'write' must return N-dimensional region"
    );

    ArgumentHandler(Write<Array<T, N>, A> arg) :
        m_array(arg.argument),
        m_access_mapper(arg.access_mapper),
        m_builder(arg.argument.sizes(), BufferLayout::for_type<T>()) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    void initialize(const TaskInit& init) {}

    type process_chunk(TaskBuilder& builder) {
        auto access_region = m_access_mapper(builder.chunk, m_builder.sizes());
        auto buffer_index = m_builder.add_chunk(builder, access_region);
        auto domain = views::domain_subbounds<N> {access_region.offset, access_region.sizes};
        return {buffer_index, domain};
    }

    void finalize(const TaskResult& result) {
        m_array = Array<T, N>(m_builder.build(result.worker, result.graph));
    }

  private:
    Array<T, N>& m_array;
    A m_access_mapper;
    ArrayBuilder<N> m_builder;
};

template<typename T, size_t N>
struct ArgumentHandler<Reduce<Array<T, N>>> {
    using type = ArrayArgument<T, views::domain_subbounds<N>>;

    ArgumentHandler(Reduce<Array<T, N>> arg) :
        m_array(arg.argument),
        m_builder(arg.argument.sizes(), DataType::of<T>(), arg.op) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    void initialize(const TaskInit& init) {}

    type process_chunk(TaskBuilder& builder) {
        auto access_region = m_builder.sizes();
        auto buffer_index = m_builder.add_chunk(builder, access_region);
        auto domain = views::domain_subbounds<N> {access_region};
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

    using type = ArrayArgument<T, views::domain_subbounds<K + N>>;

    ArgumentHandler(Reduce<Array<T, N>, All, P> arg) :
        m_array(arg.argument),
        m_builder(arg.argument.sizes(), DataType::of<T>(), arg.op),
        m_private_mapper(arg.private_mapper) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    void initialize(const TaskInit& init) {}

    type process_chunk(TaskBuilder& builder) {
        auto access_region = Range<N>(m_builder.sizes());
        auto private_region = m_private_mapper(builder.chunk);
        auto buffer_index = m_builder.add_chunk(builder, access_region, private_region.size());

        auto domain = views::domain_subbounds<K + N> {
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
    P m_private_mapper;
};

template<typename T, size_t N, typename A, typename P>
struct ArgumentHandler<Reduce<Array<T, N>, A, P>> {
    static constexpr size_t K = mapper_dimensionality<P>;
    using type = ArrayArgument<T, views::domain_subbounds<K + N>>;

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
        m_builder(arg.argument.sizes(), DataType::of<T>(), arg.op),
        m_access_mapper(arg.access_mapper),
        m_private_mapper(arg.private_mapper) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    void initialize(const TaskInit& init) {}

    type process_chunk(TaskBuilder& builder) {
        auto private_region = m_private_mapper(builder.chunk);
        auto access_region = m_access_mapper(builder.chunk, m_builder.sizes());
        size_t buffer_index = m_builder.add_chunk(builder, access_region, private_region.size());

        views::domain_subbounds<K + N> domain = {
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
    A m_access_mapper;
    P m_private_mapper;
};

}  // namespace kmm