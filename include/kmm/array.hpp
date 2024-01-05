#include <array>
#include <memory>
#include <utility>

#include "kmm/block.hpp"
#include "kmm/event.hpp"
#include "kmm/host/host.hpp"
#include "kmm/identifiers.hpp"
#include "kmm/memory.hpp"
#include "kmm/runtime.hpp"
#include "kmm/task_serialize.hpp"
#include "kmm/types.hpp"

namespace kmm {

template<typename T, size_t N = 1>
class Array {
  public:
    Array(std::array<index_t, N> sizes = {}, std::shared_ptr<Block> buffer = nullptr) :
        m_sizes(sizes),
        m_buffer(std::move(buffer)) {}

    template<
        typename... Sizes,
        std::enable_if_t<
            sizeof...(Sizes) == N && (std::is_convertible_v<Sizes, index_t> && ...),
            int> = 0>
    Array(Sizes... sizes) : m_sizes {sizes...} {}

    size_t rank() const {
        return N;
    }

    std::array<index_t, N> sizes() const {
        return m_sizes;
    }

    index_t size(size_t axis) const {
        return axis < N ? m_sizes[axis] : 1;
    }

    index_t size() const {
        return checked_product(m_sizes.begin(), m_sizes.end());
    }

    bool is_empty() const {
        return size() == 0;
    }

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

    Runtime runtime() const {
        return block()->runtime();
    }

    ArrayHeader header() const {
        return ArrayHeader::for_type<T>(size());
    }

    void synchronize() const {
        if (m_buffer) {
            m_buffer->synchronize();
        }
    }

    EventId prefetch(MemoryId memory_id, EventList dependencies = {}) const {
        if (m_buffer) {
            return m_buffer->prefetch(memory_id, std::move(dependencies));
        } else {
            return EventId::invalid();
        }
    }

  private:
    std::shared_ptr<Block> m_buffer;
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
        size_t output_index = requirements.add_output(std::move(header), rt);

        m_target = &array.inner;
        m_output_index = output_index;
        return {output_index, array.inner.sizes()};
    }

    void update(RuntimeImpl& rt, EventId event_id) {
        if (m_target) {
            auto block_id = BlockId(event_id, m_output_index);
            auto buffer = std::make_shared<Block>(rt.shared_from_this(), block_id);
            *m_target = Array<T, N>(m_target->sizes(), buffer);
        }
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

}  // namespace kmm