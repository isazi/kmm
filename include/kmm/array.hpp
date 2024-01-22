#pragma once

#include <array>
#include <memory>
#include <utility>

#include "kmm/block.hpp"
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

class Runtime;

class ArrayBase {
  public:
    ArrayBase(std::shared_ptr<Block> block = nullptr) : m_block(std::move(block)) {}
    virtual ~ArrayBase() = default;

    virtual size_t rank() const = 0;
    virtual index_t size(size_t axis) const = 0;

    bool has_block() const {
        return bool(m_block);
    }

    std::shared_ptr<Block> block() const {
        KMM_ASSERT(m_block != nullptr);
        return m_block;
    }

    BlockId id() const {
        return block()->id();
    }

    Runtime runtime() const;

    void synchronize() const {
        if (m_block) {
            m_block->synchronize();
        }
    }

    EventId prefetch(MemoryId memory_id, EventList dependencies = {}) const {
        if (m_block) {
            return m_block->prefetch(memory_id, std::move(dependencies));
        } else {
            return EventId::invalid();
        }
    }

  protected:
    std::shared_ptr<Block> m_block;
};

template<typename T, size_t N = 1>
class Array: public ArrayBase {
  public:
    using ArrayBase::size;

    Array(
        std::array<index_t, N> sizes = {},
        std::shared_ptr<Block> buffer = nullptr,
        size_t offset = 0) :
        m_sizes(sizes),
        m_offset(offset),
        ArrayBase(std::move(buffer)) {}

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
     * Returns the `N`-dimensional shape of this array.
     */
    std::array<index_t, N> sizes() const {
        return m_sizes;
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
     * Returns the total volume of this array (i.e., the total number of elements).
     *
     * @return The volume of this array.
     */
    index_t size() const {
        return checked_product(m_sizes.begin(), m_sizes.end());
    }

    /**
     * Check if the array contains no elements (i.e., if `size() == 0`)
     *
     * @return `true` if `size() == 0`, and `false` otherwise.
     */
    bool is_empty() const {
        for (size_t i = 0; i < rank(); i++) {
            if (m_sizes[i] == 0) {
                return true;
            }
        }

        return false;
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
    void read(T* dst_ptr, size_t length) const {
        if (!is_empty()) {
            m_block->read(dst_ptr, length * sizeof(T));
        }
    }

    /**
     * Read the data of this array into the provided memory location. This method blocks until
     * the data is available.
     *
     * @param v The host view where the output will be stored.
     */
    void read(view_mut<T, N> v) const {
        KMM_ASSERT(v.sizes() == m_sizes);

        if (!is_empty()) {
            m_block->read(v.data(), v.size_in_bytes());
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
    Array<index_t, sizeof...(Sizes)> reshape(Sizes... sizes) const {
        std::array<index_t, sizeof...(Sizes)> new_sizes = {checked_cast<index_t>(sizes)...};

        if (size() != checked_product(new_sizes.begin(), new_sizes.end())) {
            throw std::runtime_error("reshape failed, invalid dimensions");
        }

        return {new_sizes, m_block, m_offset};
    }

    Array<index_t> flatten() const {
        return {{size()}, m_block, m_offset};
    }

  private:
    size_t m_offset = 0;
    std::array<index_t, N> m_sizes;
};

template<typename T, size_t N>
struct SerializedArray {
    size_t buffer_index;
    size_t offset;
    std::array<index_t, N> sizes;
};

template<typename T, size_t N>
struct TaskArgumentDelegate<ExecutionSpace::Host, Array<T, N>> {
    using packed_type = SerializedArray<const T, N>;
    using unpacked_type = view<T, N>;

    static SerializedArray<const T, N> pack(
        RuntimeImpl& rt,
        TaskRequirements& reqs,
        Array<T, N> array) {
        return {
            .buffer_index = reqs.add_input(array.id(), rt),
            .offset = 0,
            .sizes = array.sizes()};
    }

    static void post_submission(
        RuntimeImpl& rt,
        EventId id,
        const Array<T, N>& array,
        SerializedArray<const T, N> arg) {}

    static view<T, N> unpack(TaskContext& context, SerializedArray<const T, N> array) {
        const auto& access = context.inputs.at(array.buffer_index);
        const auto* header = dynamic_cast<const ArrayHeader*>(access.header.get());
        KMM_ASSERT(header != nullptr && header->element_type() == typeid(T));

        const auto& alloc = dynamic_cast<const HostAllocation&>(*access.allocation);
        const auto* ptr = reinterpret_cast<const T*>(alloc.data()) + array.offset;
        return {ptr};
    }
};

template<typename T, size_t N>
struct TaskArgumentDelegate<ExecutionSpace::Host, Write<Array<T, N>>> {
    using packed_type = SerializedArray<T, N>;
    using unpacked_type = view_mut<T, N>;

    static SerializedArray<T, N> pack(
        RuntimeImpl& rt,
        TaskRequirements& reqs,
        Write<Array<T, N>> array) {
        auto header = std::make_unique<ArrayHeader>(array.inner.header());
        return {
            .buffer_index = reqs.add_output(std::move(header), rt),
            .offset = 0,
            .sizes = array.inner.sizes()};
    }

    static void post_submission(
        RuntimeImpl& rt,
        EventId id,
        const Write<Array<T, N>>& array,
        SerializedArray<T, N> arg) {
        auto block_id = BlockId(id, arg.buffer_index);
        auto block = std::make_shared<Block>(rt.shared_from_this(), block_id);
        array.inner = Array<T, N>(arg.sizes, block);
    }

    static view_mut<T, N> unpack(TaskContext& context, SerializedArray<T, N> array) {
        const auto& access = context.outputs.at(array.buffer_index);
        const auto* header = dynamic_cast<const ArrayHeader*>(access.header);
        KMM_ASSERT(header != nullptr && header->element_type() == typeid(T));

        const auto& alloc = dynamic_cast<const HostAllocation&>(*access.allocation);
        auto* ptr = reinterpret_cast<T*>(alloc.data()) + array.offset;
        return {ptr};
    }
};

#ifdef KMM_USE_CUDA

template<typename T, size_t N>
struct TaskArgumentDelegate<ExecutionSpace::Cuda, Array<T, N>> {
    using packed_type = SerializedArray<const T, N>;
    using unpacked_type = cuda_view<T, N>;

    static SerializedArray<const T, N> pack(
        RuntimeImpl& rt,
        TaskRequirements& reqs,
        Array<T, N> array) {
        return {
            .buffer_index = reqs.add_input(array.id(), rt),
            .offset = 0,
            .sizes = array.sizes()};
    }

    static void post_submission(
        RuntimeImpl& rt,
        EventId id,
        const Array<T, N>& array,
        SerializedArray<const T, N> arg) {}

    static cuda_view<T, N> unpack(TaskContext& context, SerializedArray<const T, N> array) {
        const auto& access = context.inputs.at(array.buffer_index);
        const auto* header = dynamic_cast<const ArrayHeader*>(access.header.get());
        KMM_ASSERT(header != nullptr && header->element_type() == typeid(T));

        const auto& alloc = dynamic_cast<const CudaAllocation&>(*access.allocation);
        const auto* ptr = reinterpret_cast<const T*>(alloc.data()) + array.offset;
        return {ptr};
    }
};

template<typename T, size_t N>
struct TaskArgumentDelegate<ExecutionSpace::Cuda, Write<Array<T, N>>> {
    using packed_type = SerializedArray<T, N>;
    using unpacked_type = cuda_view_mut<T, N>;

    static SerializedArray<T, N> pack(
        RuntimeImpl& rt,
        TaskRequirements& reqs,
        Write<Array<T, N>> array) {
        auto header = std::make_unique<ArrayHeader>(array.inner.header());
        return {
            .buffer_index = reqs.add_output(std::move(header), rt),
            .offset = 0,
            .sizes = array.inner.sizes()};
    }

    static void post_submission(
        RuntimeImpl& rt,
        EventId id,
        const Write<Array<T, N>>& array,
        SerializedArray<T, N> arg) {
        auto block_id = BlockId(id, arg.buffer_index);
        auto block = std::make_shared<Block>(rt.shared_from_this(), block_id);
        array.inner = Array<T, N>(arg.sizes, block);
    }

    static cuda_view_mut<T, N> unpack(TaskContext& context, SerializedArray<T, N> array) {
        const auto& access = context.outputs.at(array.buffer_index);
        const auto* header = dynamic_cast<const ArrayHeader*>(access.header);
        KMM_ASSERT(header != nullptr && header->element_type() == typeid(T));

        const auto& alloc = dynamic_cast<const CudaAllocation&>(*access.allocation);
        auto* ptr = reinterpret_cast<T*>(alloc.data()) + array.offset;
        return {ptr};
    }
};

#endif  // KMM_USE_CUDA

}  // namespace kmm