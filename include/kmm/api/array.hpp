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

    template<typename M = All>
    Access<Array<T, N>, Read<std::decay_t<M>>> access(M&& mapper = {}) {
        return {*this, {mapper}};
    }

    template<typename M = All>
    Access<const Array<T, N>, Read<std::decay_t<M>>> access(M&& mapper = {}) const {
        return {*this, {mapper}};
    }

    template<typename M>
    Access<Array<T, N>, Read<std::decay_t<M>>> operator[](M&& mapper) {
        return access(mapper);
    }

    template<typename M>
    Access<const Array<T, N>, Read<std::decay_t<M>>> operator[](M&& mapper) const {
        return access(mapper);
    }

    template<typename... Is>
    Access<Array<T, N>, Read<MultiIndexMap<N>>> operator()(const Is&... index) {
        return access(slice(index...));
    }

    template<typename... Is>
    Access<const Array<T, N>, Read<MultiIndexMap<N>>> operator()(const Is&... index) const {
        return access(slice(index...));
    }

    void copy_bytes_to(void* output, size_t num_bytes) const {
        KMM_ASSERT(num_bytes % sizeof(T) == 0);
        KMM_ASSERT(compare_equal(num_bytes / sizeof(T), size()));

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
struct ArgumentHandler<Access<const Array<T, N>, Read<All>>> {
    using type = ArrayArgument<const T, views::dynamic_domain<N>>;

    ArgumentHandler(Access<const Array<T, N>, Read<All>> access) :
        m_handle(access.argument.handle().shared_from_this()),
        m_chunk(m_handle->distribution().chunk(0)) {
        m_handle->distribution().region_to_chunk_index(access.argument.sizes()
        );  // Check if it is in-bounds
    }

    void initialize(const TaskGroupInfo& init) {}

    type process_chunk(TaskInstance& task) {
        auto buffer_index = task.add_buffer_requirement(BufferRequirement {
            .buffer_id = m_handle->buffer(0),
            .memory_id = m_chunk.owner_id,
            .access_mode = AccessMode::Read});

        auto domain = views::dynamic_domain<N> {m_chunk.size};
        return {buffer_index, domain};
    }

    void finalize(const TaskGroupResult& result) {}

  private:
    std::shared_ptr<const ArrayHandle<N>> m_handle;
    DataChunk<N> m_chunk;
};

template<typename T, size_t N>
struct ArgumentHandler<Array<T, N>>: ArgumentHandler<Access<const Array<T, N>, Read<All>>> {
    ArgumentHandler(const Array<T, N>& arg) :  //
        ArgumentHandler<Access<const Array<T, N>, Read<All>>>(  //
            Access<const Array<T, N>, Read<All>>(arg)
        ) {}
};

template<typename T, size_t N>
struct ArgumentHandler<Access<Array<T, N>, Write<All>>> {
    using type = ArrayArgument<T, views::dynamic_domain<N>>;

    ArgumentHandler(Access<Array<T, N>, Write<All>> access) :
        m_array(access.argument),
        m_builder(access.argument.sizes(), DataLayout::for_type<T>()) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    void initialize(const TaskGroupInfo& init) {}

    type process_chunk(TaskInstance& task) {
        auto access_region = m_builder.sizes();
        auto buffer_index = task.add_buffer_requirement(
            m_builder.add_chunk(task.graph, task.memory_id, access_region)
        );

        views::dynamic_domain<N> domain = {access_region};
        return {buffer_index, domain};
    }

    void finalize(const TaskGroupResult& result) {
        auto handle =
            std::make_shared<ArrayHandle<N>>(result.worker, m_builder.build(result.graph));
        m_array = Array<T, N>(handle);
    }

  private:
    Array<T, N>& m_array;
    ArrayBuilder<N> m_builder;
};

template<typename T, size_t N, typename M>
struct ArgumentHandler<Access<const Array<T, N>, Read<M>>> {
    using type = ArrayArgument<const T, views::dynamic_subdomain<N>>;

    static_assert(
        is_dimensionality_accepted_by_mapper<M, N>,
        "mapper of 'read' must return N-dimensional region"
    );

    ArgumentHandler(Access<const Array<T, N>, Read<M>> access) :
        m_handle(access.argument.handle().shared_from_this()),
        m_access_mapper(access.mode.access_mapper) {}

    void initialize(const TaskGroupInfo& init) {}

    type process_chunk(TaskInstance& task) {
        auto array_size = m_handle->distribution().array_size();
        auto access_region = m_access_mapper(task.chunk, Range<N>(array_size));
        auto chunk_index = m_handle->distribution().region_to_chunk_index(access_region);

        auto buffer_index = task.add_buffer_requirement(BufferRequirement {
            .buffer_id = m_handle->buffer(chunk_index),
            .memory_id = task.memory_id,
            .access_mode = AccessMode::Read});

        auto chunk = m_handle->distribution().chunk(chunk_index);
        auto domain = views::dynamic_subdomain<N> {chunk.offset, chunk.size};
        return {buffer_index, domain};
    }

    void finalize(const TaskGroupResult& result) {}

  private:
    std::shared_ptr<const ArrayHandle<N>> m_handle;
    M m_access_mapper;
};

template<typename T, size_t N, typename M>
struct ArgumentHandler<Access<Array<T, N>, Read<M>>>: //
    public ArgumentHandler<Access<const Array<T, N>, Read<M>>> { //
    ArgumentHandler(Access<Array<T, N>, Read<M>> access) :  //
        ArgumentHandler<Access<const Array<T, N>, Read<M>>>(access) {}
};

template<typename T, size_t N, typename M>
struct ArgumentHandler<Access<Array<T, N>, Write<M>>> {
    using type = ArrayArgument<T, views::dynamic_subdomain<N>>;

    static_assert(
        is_dimensionality_accepted_by_mapper<M, N>,
        "mapper of 'write' must return N-dimensional region"
    );

    ArgumentHandler(Access<Array<T, N>, Write<M>> access) :
        m_array(access.argument),
        m_access_mapper(access.mode.access_mapper),
        m_builder(access.argument.sizes(), DataLayout::for_type<T>()) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    void initialize(const TaskGroupInfo& init) {}

    type process_chunk(TaskInstance& task) {
        auto access_region = m_access_mapper(task.chunk, Range<N>(m_builder.sizes()));
        auto buffer_index = task.add_buffer_requirement(
            m_builder.add_chunk(task.graph, task.memory_id, access_region)
        );

        auto domain = views::dynamic_subdomain<N> {access_region.offset, access_region.sizes};
        return {buffer_index, domain};
    }

    void finalize(const TaskGroupResult& result) {
        auto handle = std::make_shared<ArrayHandle<N>>(  //
            result.worker,
            m_builder.build(result.graph)
        );

        m_array = Array<T, N>(handle);
    }

  private:
    Array<T, N>& m_array;
    M m_access_mapper;
    ArrayBuilder<N> m_builder;
};

template<typename T, size_t N>
struct ArgumentHandler<Access<Array<T, N>, Reduce<>>> {
    using type = ArrayArgument<T, views::dynamic_domain<N>>;

    ArgumentHandler(Access<Array<T, N>, Reduce<>> access) :
        m_array(access.argument),
        m_builder(access.argument.sizes(), DataType::of<T>(), access.mode.op) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    void initialize(const TaskGroupInfo& init) {}

    type process_chunk(TaskInstance& task) {
        auto access_region = m_builder.sizes();
        auto buffer_index = task.add_buffer_requirement(
            m_builder.add_chunk(task.graph, task.memory_id, access_region)
        );

        auto domain = views::dynamic_domain<N> {access_region};
        return {buffer_index, domain};
    }

    void finalize(const TaskGroupResult& result) {
        auto handle = std::make_shared<ArrayHandle<N>>(  //
            result.worker,
            m_builder.build(result.graph)
        );

        m_array = Array<T, N>(handle);
    }

  private:
    Array<T, N>& m_array;
    ArrayReductionBuilder<N> m_builder;
};

template<typename T, size_t N, typename M, typename P>
struct ArgumentHandler<Access<Array<T, N>, Reduce<M, P>>> {
    static constexpr size_t K = mapper_dimensionality<P>;
    using type = ArrayArgument<T, views::dynamic_subdomain<K + N>>;

    static_assert(
        is_dimensionality_accepted_by_mapper<M, N>,
        "mapper of 'reduce' must return N-dimensional region"
    );

    static_assert(
        is_dimensionality_accepted_by_mapper<P, K>,
        "private mapper of 'reduce' must return K-dimensional region"
    );

    ArgumentHandler(Access<Array<T, N>, Reduce<M, P>> access) :
        m_array(access.argument),
        m_builder(access.argument.sizes(), DataType::of<T>(), access.mode.op),
        m_access_mapper(access.mode.access_mapper),
        m_private_mapper(access.mode.private_mapper) {
        if (m_array.is_valid()) {
            throw std::runtime_error("array has already been written to, cannot overwrite array");
        }
    }

    void initialize(const TaskGroupInfo& init) {}

    type process_chunk(TaskInstance& task) {
        auto access_region = m_access_mapper(task.chunk, Range<N>(m_builder.sizes()));
        auto private_region = m_private_mapper(task.chunk);
        auto rep = private_region.size();
        size_t buffer_index = task.add_buffer_requirement(
            m_builder.add_chunk(task.graph, task.memory_id, access_region, rep)
        );

        views::dynamic_subdomain<K + N> domain = {
            private_region.offset.concat(access_region.offset),
            private_region.sizes.concat(access_region.sizes),
        };

        return {buffer_index, domain};
    }

    void finalize(const TaskGroupResult& result) {
        auto handle = std::make_shared<ArrayHandle<N>>(  //
            result.worker,
            m_builder.build(result.graph)
        );

        m_array = Array<T, N>(handle);
    }

  private:
    Array<T, N>& m_array;
    ArrayReductionBuilder<N> m_builder;
    M m_access_mapper;
    P m_private_mapper;
};

}  // namespace kmm