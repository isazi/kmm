#pragma once

#include <memory>
#include <stdexcept>
#include <typeinfo>
#include <vector>

#include "spdlog/spdlog.h"

#include "kmm/api/access.hpp"
#include "kmm/api/argument.hpp"
#include "kmm/api/array_argument.hpp"
#include "kmm/api/array_builder.hpp"
#include "kmm/api/array_handle.hpp"

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

    explicit Array(std::shared_ptr<ArrayHandle<N>> b) :
        m_handle(b),
        m_shape(m_handle->distribution().array_size()) {}

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
        return m_handle != nullptr;
    }

    const ArrayHandle<N>& handle() const {
        KMM_ASSERT(m_handle != nullptr);
        return *m_handle;
    }

    const DataDistribution<N>& distribution() const {
        return handle().distribution();
    }

    Size<N> chunk_size() const {
        return distribution().chunk_size();
    }

    int64_t chunk_size(size_t axis) const {
        return chunk_size().get(axis);
    }

    const Worker& worker() const {
        return handle().worker();
    }

    void synchronize() const {
        if (m_handle) {
            m_handle->synchronize();
        }
    }

    void reset() {
        m_handle = nullptr;
    }

    void copy_bytes_to(void* output, size_t num_bytes) const {
        KMM_ASSERT(num_bytes % sizeof(T) == 0 && compare_equal(num_bytes / sizeof(T), size()));
        handle().copy_bytes(output, sizeof(T));
    }

    void copy_to(T* output) const {
        handle().copy_bytes(output, sizeof(T));
    }

    template<typename I>
    void copy_to(T* output, I num_elements) const {
        KMM_ASSERT(compare_equal(num_elements, size()));
        handle().copy_bytes(output, sizeof(T));
    }

    void copy_to(std::vector<T>& output) const {
        output.resize(checked_cast<size_t>(size()));
        handle().copy_bytes(output, sizeof(T));
    }

    std::vector<T> copy() const {
        std::vector<T> output;
        copy_to(output);
        return output;
    }

  private:
    std::shared_ptr<ArrayHandle<N>> m_handle;
    Index<N> m_offset;  // Unused for now, always zero
    Size<N> m_shape;
};

template<typename T>
using Scalar = Array<T, 0>;

template<typename T, size_t N>
struct ArgumentHandler<Read<Array<T, N>>> {
    using type = ArrayArgument<const T, views::dynamic_domain<N>>;

    ArgumentHandler(Read<Array<T, N>> arg) :
        m_handle(arg.argument.handle().shared_from_this()),
        m_chunk(m_handle->distribution().chunk(0)) {
        m_handle->distribution().region_to_chunk_index(arg.argument.size()
        );  // Check if it is in-bounds
    }

    void initialize(const TaskSetInit& init) {}

    type process_chunk(TaskBuilder& builder) {
        auto buffer_index = builder.add_buffer_requirement(BufferRequirement {
            .buffer_id = m_handle->buffer(0),
            .memory_id = m_chunk.owner_id,
            .access_mode = AccessMode::Read});

        auto domain = views::dynamic_domain<N> {m_chunk.size};
        return {buffer_index, domain};
    }

    void finalize(const TaskSetResult& result) {}

  private:
    std::shared_ptr<const ArrayHandle<N>> m_handle;
    DataChunk<N> m_chunk;
};

template<typename T, size_t N>
struct ArgumentHandler<Write<Array<T, N>>> {
    using type = ArrayArgument<T, views::dynamic_domain<N>>;

    ArgumentHandler(Write<Array<T, N>> arg) :
        m_array(arg.argument),
        m_builder(arg.argument.sizes(), DataLayout::for_type<T>()) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    void initialize(const TaskSetInit& init) {}

    type process_chunk(TaskBuilder& builder) {
        auto access_region = m_builder.sizes();
        auto buffer_index = builder.add_buffer_requirement(
            m_builder.add_chunk(builder.graph, builder.memory_id, access_region)
        );

        views::dynamic_domain<N> domain = {access_region};
        return {buffer_index, domain};
    }

    void finalize(const TaskSetResult& result) {
        auto handle =
            std::make_shared<ArrayHandle<N>>(result.worker, m_builder.build(result.graph));
        m_array = Array<T, N>(handle);
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
    using type = ArrayArgument<const T, views::dynamic_subdomain<N>>;

    static_assert(
        is_dimensionality_accepted_by_mapper<A, N>,
        "mapper of 'read' must return N-dimensional region"
    );

    ArgumentHandler(Read<Array<T, N>, A> arg) :
        m_handle(arg.argument.handle().shared_from_this()),
        m_access_mapper(arg.access_mapper) {}

    void initialize(const TaskSetInit& init) {}

    type process_chunk(TaskBuilder& builder) {
        auto array_size = m_handle->distribution().array_size();
        auto access_region = m_access_mapper(builder.chunk, Range<N>(array_size));
        auto index = m_handle->distribution().region_to_chunk_index(access_region);

        auto buffer_index = builder.add_buffer_requirement(BufferRequirement {
            .buffer_id = m_handle->buffer(index),
            .memory_id = builder.memory_id,
            .access_mode = AccessMode::Read});

        auto chunk = m_handle->distribution().chunk(index);
        auto domain = views::dynamic_subdomain<N> {chunk.offset, chunk.size};
        return {buffer_index, domain};
    }

    void finalize(const TaskSetResult& result) {}

  private:
    std::shared_ptr<const ArrayHandle<N>> m_handle;
    A m_access_mapper;
};

template<typename T, size_t N, typename A>
struct ArgumentHandler<Write<Array<T, N>, A>> {
    using type = ArrayArgument<T, views::dynamic_subdomain<N>>;

    static_assert(
        is_dimensionality_accepted_by_mapper<A, N>,
        "mapper of 'write' must return N-dimensional region"
    );

    ArgumentHandler(Write<Array<T, N>, A> arg) :
        m_array(arg.argument),
        m_access_mapper(arg.access_mapper),
        m_builder(arg.argument.sizes(), DataLayout::for_type<T>()) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    void initialize(const TaskSetInit& init) {}

    type process_chunk(TaskBuilder& builder) {
        auto access_region = m_access_mapper(builder.chunk, Range<N>(m_builder.sizes()));
        auto buffer_index = builder.add_buffer_requirement(
            m_builder.add_chunk(builder.graph, builder.memory_id, access_region)
        );

        auto domain = views::dynamic_subdomain<N> {access_region.offset, access_region.sizes};
        return {buffer_index, domain};
    }

    void finalize(const TaskSetResult& result) {
        auto handle =
            std::make_shared<ArrayHandle<N>>(result.worker, m_builder.build(result.graph));
        m_array = Array<T, N>(handle);
    }

  private:
    Array<T, N>& m_array;
    A m_access_mapper;
    ArrayBuilder<N> m_builder;
};

template<typename T, size_t N>
struct ArgumentHandler<Reduce<Array<T, N>>> {
    using type = ArrayArgument<T, views::dynamic_subdomain<N>>;

    ArgumentHandler(Reduce<Array<T, N>> arg) :
        m_array(arg.argument),
        m_builder(arg.argument.sizes(), DataType::of<T>(), arg.op) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    void initialize(const TaskSetInit& init) {}

    type process_chunk(TaskBuilder& builder) {
        auto access_region = m_builder.sizes();
        auto buffer_index = builder.add_buffer_requirement(
            m_builder.add_chunk(builder.graph, builder.memory_id, access_region)
        );

        auto domain = views::dynamic_subdomain<N> {access_region};
        return {buffer_index, domain};
    }

    void finalize(const TaskSetResult& result) {
        auto handle =
            std::make_shared<ArrayHandle<N>>(result.worker, m_builder.build(result.graph));
        m_array = Array<T, N>(handle);
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

    using type = ArrayArgument<T, views::dynamic_subdomain<K + N>>;

    ArgumentHandler(Reduce<Array<T, N>, All, P> arg) :
        m_array(arg.argument),
        m_builder(arg.argument.sizes(), DataType::of<T>(), arg.op),
        m_private_mapper(arg.private_mapper) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    void initialize(const TaskSetInit& init) {}

    type process_chunk(TaskBuilder& builder) {
        auto access_region = Range<N>(m_builder.sizes());
        auto private_region = m_private_mapper(builder.chunk);
        auto rep = private_region.size();
        auto buffer_index = builder.add_buffer_requirement(
            m_builder.add_chunk(builder.graph, builder.memory_id, access_region, rep)
        );

        auto domain = views::dynamic_subdomain<K + N> {
            private_region.offset.concat(access_region.offset),
            private_region.sizes.concat(access_region.sizes),
        };

        return {buffer_index, domain};
    }

    void finalize(const TaskSetResult& result) {
        auto handle =
            std::make_shared<ArrayHandle<N>>(result.worker, m_builder.build(result.graph));
        m_array = Array<T, N>(handle);
    }

  private:
    Array<T, N>& m_array;
    ArrayReductionBuilder<N> m_builder;
    P m_private_mapper;
};

template<typename T, size_t N, typename A, typename P>
struct ArgumentHandler<Reduce<Array<T, N>, A, P>> {
    static constexpr size_t K = mapper_dimensionality<P>;
    using type = ArrayArgument<T, views::dynamic_subdomain<K + N>>;

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

    void initialize(const TaskSetInit& init) {}

    type process_chunk(TaskBuilder& builder) {
        auto private_region = m_private_mapper(builder.chunk);
        auto access_region = m_access_mapper(builder.chunk, Range<N>(m_builder.sizes()));
        auto rep = private_region.size();
        size_t buffer_index = builder.add_buffer_requirement(
            m_builder.add_chunk(builder.graph, builder.memory_id, access_region, rep)
        );

        views::dynamic_subdomain<K + N> domain = {
            private_region.offset.concat(access_region.offset),
            private_region.sizes.concat(access_region.sizes),
        };

        return {buffer_index, domain};
    }

    void finalize(const TaskSetResult& result) {
        auto handle =
            std::make_shared<ArrayHandle<N>>(result.worker, m_builder.build(result.graph));
        m_array = Array<T, N>(handle);
    }

  private:
    Array<T, N>& m_array;
    ArrayReductionBuilder<N> m_builder;
    A m_access_mapper;
    P m_private_mapper;
};

}  // namespace kmm