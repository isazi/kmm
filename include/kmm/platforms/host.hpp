#pragma once

#include <thread>

#include "kmm/executor.hpp"
#include "kmm/memory.hpp"
#include "kmm/runtime.hpp"
#include "kmm/runtime_impl.hpp"
#include "kmm/task_serialize.hpp"
#include "kmm/types.hpp"

namespace kmm {

class ParallelExecutorContext: public ExecutorContext {};

class ParallelExecutor: public Executor {
  public:
    struct Job;

    ParallelExecutor();
    ~ParallelExecutor() override;
    void submit(std::shared_ptr<Task>, TaskContext) override;
    void copy_async(
        const void* src_ptr,
        void* dst_ptr,
        size_t nbytes,
        std::unique_ptr<TransferCompletion> completion) const;

  private:
    struct Queue;
    std::shared_ptr<Queue> m_queue;
    std::thread m_thread;
};

class HostAllocation: public MemoryAllocation {
  public:
    explicit HostAllocation(size_t nbytes);

    void* data() const {
        return m_data.get();
    }

    size_t size() const {
        return m_nbytes;
    }

  private:
    size_t m_nbytes;
    std::unique_ptr<char[]> m_data;
};

class HostMemory: public Memory {
  public:
    HostMemory(
        std::shared_ptr<ParallelExecutor> executor,
        size_t max_bytes = std::numeric_limits<size_t>::max());

    std::optional<std::unique_ptr<MemoryAllocation>> allocate(DeviceId id, size_t num_bytes)
        override;

    void deallocate(DeviceId id, std::unique_ptr<MemoryAllocation> allocation) override;

    bool is_copy_possible(DeviceId src_id, DeviceId dst_id) override;

    void copy_async(
        DeviceId src_id,
        const MemoryAllocation* src_alloc,
        size_t src_offset,
        DeviceId dst_id,
        const MemoryAllocation* dst_alloc,
        size_t dst_offset,
        size_t num_bytes,
        std::unique_ptr<TransferCompletion> completion) override;

  private:
    std::shared_ptr<ParallelExecutor> m_executor;
    size_t m_bytes_remaining;
};

struct Host {
    template<typename Fun, typename... Args>
    EventId operator()(RuntimeImpl& rt, Fun&& fun, Args&&... args) const {
        return call_impl(std::index_sequence_for<Args...>(), rt, fun, args...);
    }

  private:
    template<typename Fun, size_t... Is, typename... Args>
    static EventId call_impl(
        std::index_sequence<Is...>,
        RuntimeImpl& rt,
        Fun&& fun,
        Args&&... args) {
        auto device_id = DeviceId(0);
        auto reqs = TaskRequirements(device_id);
        auto serializers =
            std::tuple<TaskArgumentSerializer<ExecutionSpace::Host, std::decay_t<Args>>...>();

        std::shared_ptr<Task> task = std::make_shared<TaskImpl<
            ExecutionSpace::Host,
            std::decay_t<Fun>,
            typename TaskArgumentSerializer<ExecutionSpace::Host, std::decay_t<Args>>::type...>>(
            std::forward<Fun>(fun),
            std::get<Is>(serializers).serialize(std::forward<Args>(args), reqs)...);

        auto event_id = rt.submit_task(std::move(task), std::move(reqs));
        (std::get<Is>(serializers).update(rt, event_id), ...);

        return event_id;
    }
};

template<typename T, size_t N>
struct SerializedArray {
    size_t buffer_index;
    std::array<index_t, N> sizes;
};

template<typename T, size_t N>
struct TaskArgumentSerializer<ExecutionSpace::Host, Array<T, N>> {
    using type = SerializedArray<const T, N>;

    type serialize(const Array<T>& array, TaskRequirements& requirements) {
        size_t index = requirements.inputs.size();
        requirements.inputs.push_back(TaskInput {
            .memory_id = requirements.device_id,
            .block_id = array.id(),
        });

        return {index, array.sizes()};
    }

    void update(RuntimeImpl& rt, EventId event_id) {}
};

template<typename T, size_t N>
struct TaskArgumentSerializer<ExecutionSpace::Host, Write<Array<T, N>>> {
    using type = SerializedArray<T, N>;

    type serialize(const Write<Array<T, N>>& array, TaskRequirements& requirements) {
        size_t output_index = requirements.outputs.size();
        auto header = array.inner.header();

        requirements.outputs.push_back(TaskOutput {
            .memory_id = requirements.device_id,
            .header = std::make_unique<decltype(header)>(header),
        });

        m_target = &array.inner;
        m_output_index = output_index;

        return {output_index, array.inner.sizes()};
    }

    void update(RuntimeImpl& rt, EventId event_id) {
        if (m_target) {
            auto buffer = Buffer(BlockId(event_id, m_output_index), rt.shared_from_this());
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