#include "block.hpp"
#include "block_header.hpp"
#include "runtime.hpp"
#include "task_serialize.hpp"

namespace kmm {

template<typename T>
class Scalar {
  public:
    Scalar(std::shared_ptr<Block> block = nullptr) : m_block(block) {}

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

    Runtime runtime() const {
        return block()->runtime();
    }

  private:
    std::shared_ptr<Block> m_block;
};

template<typename T>
struct SerializedScalar {
    size_t buffer_index;
};

template<ExecutionSpace Space, typename T>
struct TaskArgumentSerializer<Space, Scalar<T>> {
    using type = SerializedScalar<const T>;

    type serialize(RuntimeImpl& rt, const Scalar<T>& scalar, TaskRequirements& requirements) {
        return {requirements.add_input(scalar.id())};
    }

    void update(RuntimeImpl& rt, EventId event_id) {}
};

template<ExecutionSpace Space, typename T>
struct TaskArgumentSerializer<Space, Write<Scalar<T>>> {
    using type = SerializedScalar<T>;

    type serialize(
        RuntimeImpl& rt,
        const Write<Scalar<T>>& scalar,
        TaskRequirements& requirements) {
        auto header = std::make_unique<ScalarHeader<T>>();
        m_output_index = requirements.add_output(std::move(header), rt);
        m_target = &scalar.inner;

        return {m_output_index};
    }

    void update(RuntimeImpl& rt, EventId event_id) {
        if (m_target) {
            auto block_id = BlockId(event_id, m_output_index);
            auto block = std::make_shared<Block>(rt.shared_from_this(), block_id);
            *m_target = Scalar<T>(std::move(block));
        }
    }

  private:
    Scalar<T>* m_target = nullptr;
    size_t m_output_index = ~0;
};

template<ExecutionSpace Space, typename T>
struct TaskArgumentDeserializer<Space, SerializedScalar<T>> {
    using type = T&;

    type deserialize(const SerializedScalar<T>& scalar, TaskContext& context) {
        const auto& access = context.outputs.at(scalar.buffer_index);
        auto* header = dynamic_cast<ScalarHeader<T>*>(access.header);
        KMM_ASSERT(header != nullptr);

        return header->get();
    }
};

template<ExecutionSpace Space, typename T>
struct TaskArgumentDeserializer<Space, SerializedScalar<const T>> {
    using type = const T&;

    type deserialize(const SerializedScalar<const T>& scalar, TaskContext& context) {
        const auto& access = context.inputs.at(scalar.buffer_index);
        const auto* header = dynamic_cast<const ScalarHeader<T>*>(access.header.get());
        KMM_ASSERT(header != nullptr);

        return header->get();
    }
};

}  // namespace kmm