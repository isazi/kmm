#pragma once

#include <array>
#include <memory>
#include <utility>

#include "checked_math.hpp"

#include "kmm/block.hpp"
#include "kmm/block_header.hpp"
#include "kmm/cuda/memory.hpp"
#include "kmm/event_list.hpp"
#include "kmm/host/memory.hpp"
#include "kmm/identifiers.hpp"
#include "kmm/memory.hpp"
#include "kmm/task_serialize.hpp"
#include "kmm/utils.hpp"

namespace kmm {

class Runtime;

class ArrayBase {
  public:
    ArrayBase(std::shared_ptr<Block> block = nullptr) : m_buffer(std::move(block)) {}
    virtual ~ArrayBase() = default;

    virtual ArrayHeader header() const = 0;
    virtual size_t rank() const = 0;
    virtual index_t size(size_t axis) const = 0;

    bool has_block() const {
        return bool(m_buffer);
    }

    std::shared_ptr<Block> block() const {
        KMM_ASSERT(m_buffer != nullptr);
        return m_buffer;
    }

    BlockId id() const {
        return block()->id();
    }

    Runtime runtime() const;

    void synchronize() const {
        if (m_buffer) {
            m_buffer->synchronize();
        }
    }

    index_t size() const {
        index_t volume = 1;
        for (size_t i = 0; i < rank(); i++) {
            volume = checked_mul(volume, size(i));
        }

        return volume;
    }

    bool is_empty() const {
        for (size_t i = 0; i < rank(); i++) {
            if (size(i) == 0) {
                return true;
            }
        }

        return false;
    }

    EventId prefetch(MemoryId memory_id, EventList dependencies = {}) const {
        if (m_buffer) {
            return m_buffer->prefetch(memory_id, std::move(dependencies));
        } else {
            return EventId::invalid();
        }
    }

  protected:
    std::shared_ptr<Block> m_buffer;
};

template<typename T, size_t N = 1>
class Array: public ArrayBase {
  public:
    using ArrayBase::size;

    Array(std::array<index_t, N> sizes = {}, std::shared_ptr<Block> buffer = nullptr) :
        m_sizes(sizes),
        ArrayBase(std::move(buffer)) {}

    template<
        typename... Sizes,
        std::enable_if_t<
            sizeof...(Sizes) == N && (std::is_convertible_v<Sizes, index_t> && ...),
            int> = 0>
    Array(Sizes... sizes) : m_sizes {sizes...} {}

    size_t rank() const final {
        return N;
    }

    index_t size(size_t axis) const final {
        return axis < N ? m_sizes[axis] : 1;
    }

    std::array<index_t, N> sizes() const {
        return m_sizes;
    }

    ArrayHeader header() const final {
        return ArrayHeader::for_type<T>(size());
    }

    void read(T* dst_ptr, size_t length) const {
        m_buffer->read(dst_ptr, length * sizeof(T));
    }

    std::vector<T> read() const {
        auto length = checked_cast<size_t>(size());
        std::vector<T> buffer;
        buffer.resize(length);

        read(buffer.data(), buffer.size());
        return buffer;
    }

  private:
    std::array<index_t, N> m_sizes;
};

template<typename T, size_t N>
struct SerializedArray {
    size_t buffer_index;
    std::array<index_t, N> sizes;
};

template<ExecutionSpace Space, typename T, size_t N>
struct TaskArgumentSerializer<Space, Array<T, N>> {
    using type = SerializedArray<const T, N>;

    type serialize(RuntimeImpl& rt, const Array<T>& array, TaskRequirements& requirements) {
        size_t index = requirements.add_input(array.id(), rt);
        return {index, array.sizes()};
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
        return {m_output_index, array.inner.sizes()};
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
        return reinterpret_cast<T*>(alloc.data());
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
        return reinterpret_cast<const T*>(alloc.data());
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

        auto& alloc = dynamic_cast<const CudaAllocation&>(*access.allocation);
        return reinterpret_cast<T*>(alloc.data());
    }
};

template<typename T, size_t N>
struct TaskArgumentDeserializer<ExecutionSpace::Cuda, SerializedArray<const T, N>> {
    using type = const T*;

    const T* deserialize(const SerializedArray<const T, N>& array, TaskContext& context) {
        const auto& access = context.inputs.at(array.buffer_index);
        const auto* header = dynamic_cast<const ArrayHeader*>(access.header.get());
        KMM_ASSERT(header != nullptr && header->element_type() == typeid(T));

        auto& alloc = dynamic_cast<const CudaAllocation&>(*access.allocation);
        return reinterpret_cast<const T*>(alloc.data());
    }
};
#endif // KMM_USE_CUDA

}  // namespace kmm