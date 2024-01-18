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
#include "kmm/task_serialize.hpp"
#include "kmm/utils/checked_math.hpp"

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
            KMM_TODO();
        }

        return {new_sizes, m_block, m_offset};
    }

    Array<index_t> flatten() const {
        return reshape(size());
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

template<ExecutionSpace Space, typename T, size_t N>
struct TaskArgumentSerializer<Space, Array<T, N>> {
    using type = SerializedArray<const T, N>;

    type serialize(RuntimeImpl& rt, const Array<T>& array, TaskRequirements& requirements) {
        size_t index = requirements.add_input(array.id(), rt);
        return {index, 0, array.sizes()};
    }

    void update(RuntimeImpl& rt, EventId event_id) {}
};

template<ExecutionSpace Space, typename T, size_t N>
struct TaskArgumentSerializer<Space, Write<Array<T, N>>> {
    using type = SerializedArray<T, N>;

    type serialize(
        RuntimeImpl& rt,
        const Write<Array<T, N>>& array,
        TaskRequirements& requirements) {
        auto header = std::make_unique<ArrayHeader>(array.inner.header());

        m_target = &array.inner;
        m_output_index = requirements.add_output(std::move(header), rt);
        return {m_output_index, 0, array.inner.sizes()};
    }

    void update(RuntimeImpl& rt, EventId event_id) {
        if (m_target) {
            auto block_id = BlockId(event_id, m_output_index);
            auto buffer = std::make_shared<Block>(rt.shared_from_this(), block_id);
            *m_target = Array<T, N>(m_target->sizes(), buffer);
        }

        m_target = nullptr;
    }

  private:
    Array<T, N>* m_target = nullptr;
    size_t m_output_index = ~0;
};

template<typename T, size_t N>
struct TaskArgumentDeserializer<ExecutionSpace::Host, SerializedArray<T, N>> {
    using type = T*;

    T* deserialize(const SerializedArray<T, N>& array, TaskContext& context) {
        const auto& access = context.outputs.at(array.buffer_index);
        const auto* header = dynamic_cast<const ArrayHeader*>(access.header);
        KMM_ASSERT(header != nullptr && header->element_type() == typeid(T));

        auto& alloc = dynamic_cast<const HostAllocation&>(*access.allocation);
        return reinterpret_cast<T*>(alloc.data()) + array.offset;
    }
};

template<typename T, size_t N>
struct TaskArgumentDeserializer<ExecutionSpace::Host, SerializedArray<const T, N>> {
    using type = const T*;

    const T* deserialize(const SerializedArray<const T, N>& array, TaskContext& context) {
        const auto& access = context.inputs.at(array.buffer_index);
        const auto* header = dynamic_cast<const ArrayHeader*>(access.header.get());
        KMM_ASSERT(header != nullptr && header->element_type() == typeid(T));

        auto& alloc = dynamic_cast<const HostAllocation&>(*access.allocation);
        return reinterpret_cast<const T*>(alloc.data()) + array.offset;
    }
};

#ifdef KMM_USE_CUDA
template<typename T, size_t N>
struct TaskArgumentDeserializer<ExecutionSpace::Cuda, SerializedArray<T, N>> {
    using type = T*;

    T* deserialize(const SerializedArray<T, N>& array, TaskContext& context) {
        const auto& access = context.outputs.at(array.buffer_index);
        const auto* header = dynamic_cast<const ArrayHeader*>(access.header);
        KMM_ASSERT(header != nullptr && header->element_type() == typeid(T));

        const auto& alloc = dynamic_cast<const CudaAllocation&>(*access.allocation);
        return reinterpret_cast<T*>(alloc.data()) + array.offset;
    }
};

template<typename T, size_t N>
struct TaskArgumentDeserializer<ExecutionSpace::Cuda, SerializedArray<const T, N>> {
    using type = const T*;

    const T* deserialize(const SerializedArray<const T, N>& array, TaskContext& context) {
        const auto& access = context.inputs.at(array.buffer_index);
        const auto* header = dynamic_cast<const ArrayHeader*>(access.header.get());
        KMM_ASSERT(header != nullptr && header->element_type() == typeid(T));

        const auto& alloc = dynamic_cast<const CudaAllocation&>(*access.allocation);
        return reinterpret_cast<const T*>(alloc.data()) + array.offset;
    }
};
#endif  // KMM_USE_CUDA

}  // namespace kmm