#include "block.hpp"
#include "block_header.hpp"
#include "runtime.hpp"
#include "task_serialize.hpp"

namespace kmm {

class FutureBase {
  public:
    FutureBase(const std::type_info* type = nullptr, std::shared_ptr<Block> block = nullptr) :
        m_type(type),
        m_block(std::move(block)) {}

    bool has_block() const;
    std::shared_ptr<Block> block() const;
    BlockId id() const;
    Runtime runtime() const;
    void synchronize() const;
    bool is(const std::type_info&) const;

    operator bool() const {
        return has_block();
    }

    template<typename T>
    bool is() const {
        return is(typeid(T));
    }

  private:
    const std::type_info* m_type;
    std::shared_ptr<Block> m_block;
};

template<typename T>
class Future: public FutureBase {
  public:
    Future(std::shared_ptr<Block> block = nullptr) : FutureBase(&typeid(T), block) {}

    std::shared_ptr<const T> wait() const {
        auto header = std::dynamic_pointer_cast<ScalarHeader<T>>(block()->header());
        const T* ptr = header->get();
        return {std::move(header), ptr};
    }
};

template<typename T>
struct SerializedFuture {
    size_t buffer_index;
};

template<ExecutionSpace Space, typename T>
struct TaskArgumentSerializer<Space, Future<T>> {
    using type = SerializedFuture<const T>;

    type serialize(RuntimeImpl& rt, const Future<T>& scalar, TaskRequirements& requirements) {
        return {requirements.add_input(scalar.id())};
    }

    void update(RuntimeImpl& rt, EventId event_id) {}
};

template<ExecutionSpace Space, typename T>
struct TaskArgumentSerializer<Space, Write<Future<T>>> {
    using type = SerializedFuture<T>;

    type serialize(
        RuntimeImpl& rt,
        const Write<Future<T>>& scalar,
        TaskRequirements& requirements) {
        auto header = std::make_unique<ScalarHeader<T>>();
        m_target = &scalar.inner;
        m_output_index = requirements.add_output(std::move(header), rt);

        return {m_output_index};
    }

    void update(RuntimeImpl& rt, EventId event_id) {
        if (m_target) {
            auto block_id = BlockId(event_id, m_output_index);
            auto block = std::make_shared<Block>(rt.shared_from_this(), block_id);
            *m_target = Future<T>(std::move(block));
        }
    }

  private:
    Future<T>* m_target = nullptr;
    size_t m_output_index = ~0;
};

template<ExecutionSpace Space, typename T>
struct TaskArgumentDeserializer<Space, SerializedFuture<T>> {
    using type = T&;

    type deserialize(const SerializedFuture<T>& scalar, TaskContext& context) {
        const auto& access = context.outputs.at(scalar.buffer_index);
        auto* header = dynamic_cast<ScalarHeader<T>*>(access.header);
        KMM_ASSERT(header != nullptr);

        return header->get();
    }
};

template<ExecutionSpace Space, typename T>
struct TaskArgumentDeserializer<Space, SerializedFuture<const T>> {
    using type = const T&;

    type deserialize(const SerializedFuture<const T>& scalar, TaskContext& context) {
        const auto& access = context.inputs.at(scalar.buffer_index);
        const auto* header = dynamic_cast<const ScalarHeader<T>*>(access.header.get());
        KMM_ASSERT(header != nullptr);

        return header->get();
    }
};

}  // namespace kmm