#pragma once

#include "kmm/api/array_base.hpp"
#include "kmm/api/packed_array.hpp"

namespace kmm {

template<typename T, size_t N = 1>
class Array: public ArrayBase {
  public:
    Array(dim<N> shape = {}) : m_shape(shape) {}
    Array(std::shared_ptr<ArrayImpl<N>> impl) : m_shape(impl->m_global_size), m_impl(impl) {}

    template<
        typename... Sizes,
        std::enable_if_t<
            sizeof...(Sizes) == N && (std::is_convertible_v<Sizes, int64_t> && ...),
            int> = 0>
    Array(Sizes... sizes) : Array {dim<N> {checked_cast<int64_t>(sizes)...}} {}

    size_t rank() const final {
        return N;
    }

    size_t size(size_t axis) const final {
        return m_shape.get(axis);
    }

    size_t chunk_size(size_t axis) const final {
        return chunk_size().get(axis);
    }

    size_t num_chunks() const final {
        return m_impl ? m_impl->m_buffers.size() : 0;
    }

    std::shared_ptr<Buffer> chunk(size_t index) const final {
        KMM_ASSERT(index < num_chunks());
        return m_impl->m_buffers[index];
    }

    dim<N> shape() const {
        return m_shape;
    }

    dim<N> chunk_size() const {
        return m_impl ? m_impl->m_chunk_size : dim<N>();
    }

    bool is_present() const {
        return m_impl != nullptr;
    }

    const ArrayImpl<N>& inner() const {
        KMM_ASSERT(m_impl != nullptr);
        return *m_impl;
    }

  private:
    dim<N> m_shape;
    std::shared_ptr<ArrayImpl<N>> m_impl;
};

template<typename T, size_t N, typename P>
struct TaskDataProcessor<Read<Array<T, N>, P>> {
    using type = PackedArray<const T, N>;

    TaskDataProcessor(Read<Array<T, N>, P> arg) :
        m_array(arg.inner.inner()),
        m_shape(arg.inner.shape()),
        m_index_mapping(arg.index_mapping) {}

    template<size_t M>
    type pre_enqueue(Chunk<M> chunk, TaskRequirements& req) {
        auto region = m_index_mapping(chunk, m_shape);
        auto c = m_array.find_chunk(region);

        auto buffer_index = req.inputs.size();
        req.inputs.push_back(c.buffer);

        return {.buffer_index = buffer_index, .offset = c.offset, .sizes = c.shape};
    }

    template<size_t M>
    void post_enqueue(Chunk<M> chunk, TaskResult& result) {}
    void finalize() {}

  private:
    const ArrayImpl<N>& m_array;
    dim<N> m_shape;
    P m_index_mapping;
};

template<typename T, size_t N, typename P>
struct TaskDataProcessor<Write<Array<T, N>, P>> {
    using type = PackedArray<T, N>;

    TaskDataProcessor(Write<Array<T, N>, P> arg) :
        m_array(arg.inner),
        m_shape(arg.inner.shape()),
        m_index_mapping(arg.index_mapping) {}

    template<size_t M>
    type pre_enqueue(Chunk<M> chunk, TaskRequirements& req) {
        auto region = m_index_mapping(chunk, m_shape);

        auto layout = BufferLayout::for_type<T>(region.size());
        req.outputs.push_back(layout);

        m_chunks.push_back(
            ArrayChunk<N> {.offset = region.offset, .shape = region.sizes, .buffer = nullptr});

        return {
            .buffer_index = m_prev_output_index,
            .offset = region.offset,
            .sizes = region.sizes};
    }

    template<size_t M>
    void post_enqueue(Chunk<M> chunk, TaskResult& result) {
        m_chunks.back().buffer = result.outputs.at(m_prev_output_index);
    }

    void finalize() {
        m_array = std::make_shared<ArrayImpl<N>>(m_shape, m_chunks);
    }

  private:
    Array<T, N>& m_array;
    dim<N> m_shape;
    P m_index_mapping;
    std::vector<ArrayChunk<N>> m_chunks;
    size_t m_prev_output_index = ~0;
};

template<typename T, size_t N>
struct TaskDataProcessor<Array<T, N>>: TaskDataProcessor<Read<Array<T, N>>> {
    TaskDataProcessor(Array<T, N> value) : TaskDataProcessor<Read<Array<T, N>>>(read(value)) {}
};

}  // namespace kmm