#include "block.hpp"
#include "block_header.hpp"
#include "runtime.hpp"
#include "task_argument.hpp"

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

struct SerializedFuture {
    size_t buffer_index;
};

template<ExecutionSpace Space, typename T>
struct TaskArgumentTrait<Space, Future<T>> {
    using packed_type = SerializedFuture;
    using unpacked_type = const T&;

    static packed_type pack(RuntimeImpl& rt, TaskRequirements& reqs, Future<T> future) {
        return {.buffer_index = reqs.add_input(future.id())};
    }

    static void post_submission(
        RuntimeImpl& rt,
        EventId id,
        const Future<T>& future,
        packed_type arg) {}

    static unpacked_type unpack(TaskContext& context, packed_type future) {
        const auto& access = context.inputs.at(future.buffer_index);
        const auto* header = dynamic_cast<const ScalarHeader<T>*>(access.header.get());
        KMM_ASSERT(header != nullptr);

        return header->get();
    }
};

template<ExecutionSpace Space, typename T>
struct TaskArgumentTrait<Space, Write<Future<T>>> {
    using packed_type = SerializedFuture;
    using unpacked_type = T&;

    static packed_type pack(RuntimeImpl& rt, TaskRequirements& reqs, Write<Future<T>> future) {
        return {.buffer_index = reqs.add_output(future->id())};
    }

    static void post_submission(
        RuntimeImpl& rt,
        EventId id,
        const Write<Future<T>>& future,
        packed_type arg) {
        auto block_id = BlockId(id, arg.buffer_index);
        auto block = std::make_shared<Block>(rt.shared_from_this(), block_id);
        *future = Future<T>(std::move(block));
    }

    static unpacked_type unpack(TaskContext& context, packed_type future) {
        const auto& access = context.outputs.at(future.buffer_index);
        const auto* header = dynamic_cast<ScalarHeader<T>*>(access.header);
        KMM_ASSERT(header != nullptr);

        return header->get();
    }
};

}  // namespace kmm