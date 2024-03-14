#pragma once

#include <array>
#include <memory>
#include <utility>

#include "kmm/array_base.hpp"
#include "kmm/block_header.hpp"
#include "kmm/cuda/memory.hpp"
#include "kmm/event_list.hpp"
#include "kmm/host/memory.hpp"
#include "kmm/identifiers.hpp"
#include "kmm/memory.hpp"
#include "kmm/task_argument.hpp"
#include "kmm/utils/checked_math.hpp"
#include "kmm/utils/view.hpp"

namespace kmm {

class RuntimeHandle;

template<typename T, size_t N = 1>
class Array final: public ArrayBase {
  public:
    using ArrayBase::size;

    Array(
        std::array<index_t, N> sizes = {},
        std::shared_ptr<Block> buffer = nullptr,
        size_t offset = 0) :
        ArrayBase(std::move(buffer)),
        m_sizes(sizes),
        m_offset(offset) {}

    template<
        typename... Sizes,
        std::enable_if_t<
            sizeof...(Sizes) == N && (std::is_convertible_v<Sizes, index_t> && ...),
            int> = 0>
    Array(Sizes... sizes) : m_sizes {sizes...} {}

    /**
     * Returns the rank (i.e., the number of dimensions).
     */
    size_t rank() const final {
        return N;
    }

    /**
     * Returns the number of elements along the provided `axis`.
     *
     * @param axis The target axis.
     * @return The dimensionality along the provided axis.
     */
    index_t size(size_t axis) const final {
        return axis < N ? m_sizes[axis] : 1;
    }

    /**
     * Returns the `N`-dimensional shape of this array.
     */
    std::array<index_t, N> sizes() const {
        return m_sizes;
    }

    ArrayHeader header() const {
        return ArrayHeader::for_type<T>(size());
    }

    /**
     * Read the data of this array into the provided memory location. This method blocks until
     * the data is available.
     *
     * @param dst_ptr The memory location where the data will be written.
     * @param length The size of the memory location in number of elements. Must equal `size()`.
     */
    template<typename I>
    void read(T* dst_ptr, I length) const {
        KMM_ASSERT(checked_cast<index_t>(length) == size());

        if (!is_empty()) {
            read_bytes(dst_ptr, checked_mul(checked_cast<size_t>(length), sizeof(T)));
        }
    }

    /**
     * Read the data of this array into the provided memory location. This method blocks until
     * the data is available.
     *
     * @param v The host view where the output will be stored.
     */
    void read(view_mut<T, N> v) const {
        for (size_t i = 0; i < N; i++) {
            KMM_ASSERT(v.size(i) == m_sizes[i]);
        }

        if (!is_empty()) {
            read_bytes(v.data(), v.size_in_bytes());
        }
    }

    /**
     * Read the data of this array into a `std::vector<T>`. This methods blocks until
     * the data is available.
     *
     * @return The data of this array.
     */
    std::vector<T> read() const {
        auto buffer = std::vector<T>(checked_cast<size_t>(size()));
        read(buffer.data(), buffer.size());
        return buffer;
    }

    template<typename... Sizes>
    Array<T, sizeof...(Sizes)> reshape(Sizes... sizes) const {
        std::array<index_t, sizeof...(Sizes)> new_sizes = {checked_cast<index_t>(sizes)...};

        if (size() != checked_product(new_sizes.begin(), new_sizes.end())) {
            throw std::runtime_error("reshape failed, invalid dimensions");
        }

        return {new_sizes, m_block, m_offset};
    }

    Array<T> flatten() const {
        return {{size()}, m_block, m_offset};
    }

  private:
    std::array<index_t, N> m_sizes;
    size_t m_offset = 0;
};

template<typename T, size_t N = 1>
struct PackedArray {
    size_t buffer_index;
    size_t offset;
    std::array<index_t, N> sizes;
};

template<typename I, size_t N>
static fixed_array<I, N> to_fixed_array(const std::array<I, N>& input) {
    fixed_array<I, N> result;
    for (size_t i = 0; i < N; i++) {
        result[i] = input[i];
    }
    return result;
}

template<typename T, size_t N>
struct TaskArgumentPack<ExecutionSpace::Host, Array<T, N>> {
    using type = PackedArray<const T, N>;

    static PackedArray<const T, N> pack(TaskBuilder& builder, Array<T, N> array) {
        return {//
                .buffer_index = builder.add_input(array.id()),
                .offset = 0,
                .sizes = array.sizes()};
    }
};

template<typename T, size_t N>
struct TaskArgumentUnpack<ExecutionSpace::Host, PackedArray<const T, N>> {
    using type = view<T, N>;

    static view<T, N> unpack(TaskContext& context, PackedArray<const T, N> array) {
        const auto& access = context.inputs.at(array.buffer_index);
        const auto* header = dynamic_cast<const ArrayHeader*>(access.header.get());
        KMM_ASSERT(header != nullptr && header->element_type() == typeid(T));

        const auto& alloc = dynamic_cast<const HostAllocation&>(*access.allocation);
        const auto* ptr = reinterpret_cast<const T*>(alloc.data()) + array.offset;
        return {ptr, to_fixed_array(array.sizes)};
    }
};

template<typename T, size_t N>
struct TaskArgumentPack<ExecutionSpace::Host, Write<Array<T, N>>> {
    using type = PackedArray<T, N>;

    static PackedArray<T, N> pack(TaskBuilder& builder, Write<Array<T, N>> array) {
        auto header = std::make_unique<ArrayHeader>(array->header());
        auto buffer_index = builder.add_output(std::move(header));
        auto rt = builder.runtime();

        builder.after_submission([=](EventId event_id) {
            auto block_id = BlockId(event_id, buffer_index);
            auto block = std::make_shared<Block>(rt, block_id);
            *array = Array<T, N>(array->sizes(), block);
        });

        return {//
                .buffer_index = buffer_index,
                .offset = 0,
                .sizes = array->sizes()};
    }
};

template<typename T, size_t N>
struct TaskArgumentUnpack<ExecutionSpace::Host, PackedArray<T, N>> {
    using type = view_mut<T, N>;

    static view_mut<T, N> unpack(TaskContext& context, PackedArray<T, N> array) {
        const auto& access = context.outputs.at(array.buffer_index);
        const auto* header = dynamic_cast<const ArrayHeader*>(access.header);
        KMM_ASSERT(header != nullptr && header->element_type() == typeid(T));

        const auto& alloc = dynamic_cast<const HostAllocation&>(*access.allocation);
        auto* ptr = reinterpret_cast<T*>(alloc.data()) + array.offset;
        return {ptr, to_fixed_array(array.sizes)};
    }
};

#ifdef KMM_USE_CUDA

template<typename T, size_t N>
struct TaskArgumentPack<ExecutionSpace::Cuda, Array<T, N>> {
    using type = PackedArray<const T, N>;

    static PackedArray<const T, N> pack(TaskBuilder& builder, Array<T, N> array) {
        return {//
                .buffer_index = builder.add_input(array.id()),
                .offset = 0,
                .sizes = array.sizes()};
    }
};

template<typename T, size_t N>
struct TaskArgumentUnpack<ExecutionSpace::Cuda, PackedArray<const T, N>> {
    using type = cuda_view<T, N>;

    static cuda_view<T, N> unpack(TaskContext& context, PackedArray<const T, N> array) {
        const auto& access = context.inputs.at(array.buffer_index);
        const auto* header = dynamic_cast<const ArrayHeader*>(access.header.get());
        KMM_ASSERT(header != nullptr && header->element_type() == typeid(T));

        const auto& alloc = dynamic_cast<const CudaAllocation&>(*access.allocation);
        const auto* ptr = reinterpret_cast<const T*>(alloc.data()) + array.offset;
        return {ptr, to_fixed_array(array.sizes)};
    }
};

template<typename T, size_t N>
struct TaskArgumentPack<ExecutionSpace::Cuda, Write<Array<T, N>>> {
    using type = PackedArray<T, N>;

    static PackedArray<T, N> pack(TaskBuilder& builder, Write<Array<T, N>> array) {
        auto header = std::make_unique<ArrayHeader>(array->header());
        auto buffer_index = builder.add_output(std::move(header));
        auto rt = builder.runtime();

        builder.after_submission([=](EventId event_id) {
            auto block_id = BlockId(event_id, checked_cast<uint8_t>(buffer_index));
            auto block = std::make_shared<Block>(rt, block_id);
            *array = Array<T, N>(array->sizes(), block);
        });

        return {//
                .buffer_index = buffer_index,
                .offset = 0,
                .sizes = array->sizes()};
    }
};

template<typename T, size_t N>
struct TaskArgumentUnpack<ExecutionSpace::Cuda, PackedArray<T, N>> {
    using type = cuda_view_mut<T, N>;

    static cuda_view_mut<T, N> unpack(TaskContext& context, PackedArray<T, N> array) {
        const auto& access = context.outputs.at(array.buffer_index);
        const auto* header = dynamic_cast<const ArrayHeader*>(access.header);
        KMM_ASSERT(header != nullptr && header->element_type() == typeid(T));

        const auto& alloc = dynamic_cast<const CudaAllocation&>(*access.allocation);
        auto* ptr = reinterpret_cast<T*>(alloc.data()) + array.offset;
        return {ptr, to_fixed_array(array.sizes)};
    }
};

#endif  // KMM_USE_CUDA

}  // namespace kmm
