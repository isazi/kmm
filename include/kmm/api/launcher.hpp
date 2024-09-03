#pragma once

#include "kmm/api/task_data.hpp"

namespace kmm {

template<typename F, typename... Args>
class HostTaskImpl: public HostTask {
  public:
    HostTaskImpl(F fun, std::tuple<Args...> args) : m_fun(fun), m_args(std::move(args)) {}

    void execute(TaskContext context) override {
        execute_impl(std::index_sequence_for<Args...>(), context);
    }

    template<size_t... Is>
    void execute_impl(std::index_sequence<Is...>, TaskContext& context) {
        m_fun(TaskDataDeserialize<ExecutionSpace::Host, Args>::unpack(
            context,
            std::get<Is>(m_args))...);
    }

  private:
    F m_fun;
    std::tuple<Args...> m_args;
};

struct Host {
    template<size_t N, typename F, typename... Args>
    EventId operator()(TaskBuilder& builder, Chunk<N> chunk, F fun, Args... args) {
        return builder.graph.insert_host_task(
            std::make_shared<HostTaskImpl<F, Args...>>(fun, std::make_tuple(args...)),
            builder.buffers,
            builder.dependencies);
    }
};

template<typename F, typename... Args>
class DeviceTaskImpl: public DeviceTask {
  public:
    DeviceTaskImpl(F fun, std::tuple<Args...> args) : m_fun(fun), m_args(std::move(args)) {}

    void execute(CudaDevice& device, TaskContext context) override {
        execute_impl(std::index_sequence_for<Args...>(), device, context);
    }

    template<size_t... Is>
    void execute_impl(std::index_sequence<Is...>, CudaDevice& device, TaskContext& context) {
        m_fun(
            device,
            TaskDataDeserialize<ExecutionSpace::Cuda, Args>::unpack(
                context,
                std::get<Is>(m_args))...);
    }

  private:
    F m_fun;
    std::tuple<Args...> m_args;
};

struct Cuda {
    template<size_t N, typename F, typename... Args>
    EventId operator()(TaskBuilder& builder, Chunk<N> chunk, F fun, Args... args) {
        return builder.graph.insert_device_task(
            chunk.owner_id.as_device(),
            std::make_shared<DeviceTaskImpl<F, Args...>>(fun, std::make_tuple(args...)),
            builder.buffers,
            builder.dependencies);
    }
};

struct CudaKernel {
    dim3 block_size;
    dim3 elements_per_thread;
    uint32_t shared_memory;

    CudaKernel(dim3 block_size) : CudaKernel(block_size, block_size) {}

    CudaKernel(dim3 block_size, dim3 elements_per_thread, uint32_t shared_memory = 0) :
        block_size(block_size),
        elements_per_thread(elements_per_thread),
        shared_memory(shared_memory) {}

    template<size_t N, typename... Args>
    EventId operator()(TaskBuilder& builder, Chunk<N> chunk, Args... args) {
        auto fun = [=](CudaDevice& device, auto&& kernel, auto&&... kargs) {
            device.launch(
                block_size,
                block_size,
                shared_memory,
                kernel,
                rect<N> {chunk.offset, chunk.size},
                kargs...);
        };

        return builder.graph.insert_device_task(
            chunk.owner_id.as_device(),
            std::make_shared<DeviceTaskImpl<decltype(fun), Args...>>(fun, std::make_tuple(args...)),
            builder.buffers,
            builder.dependencies);
    }
};

namespace detail {
template<size_t... Is, size_t N, typename Launcher, typename... Args>
EventId parallel_submit_impl(
    std::index_sequence<Is...>,
    Worker& worker,
    const Partition<N>& partition,
    Launcher launcher,
    Args&&... args) {
    return worker.with_task_graph([&](TaskGraph& graph) {
        EventList events;
        std::tuple<TaskDataProcessor<typename std::decay<Args>::type>...> processors = {
            std::forward<Args>(args)...};

        for (const Chunk<N>& chunk : partition.chunks) {
            TaskBuilder builder {
                .graph = graph,
                .worker = worker,
                .memory_id = chunk.owner_id,
                .buffers = {},
                .dependencies = {}};

            EventId id =
                launcher(builder, chunk, std::get<Is>(processors).process_chunk(chunk, builder)...);

            events.push_back(id);
        }

        auto result = TaskResult {.graph = graph, .worker = worker, .events = std::move(events)};

        (std::get<Is>(processors).finalize(result), ...);
        return graph.join_events(result.events);
    });
}
}  // namespace detail

template<size_t N, typename Launcher, typename... Args>
EventId parallel_submit(
    Worker& worker,
    const Partition<N>& partition,
    Launcher launcher,
    Args&&... args) {
    return detail::parallel_submit_impl(
        std::index_sequence_for<Args...> {},
        worker,
        partition,
        launcher,
        std::forward<Args>(args)...);
}

}  // namespace kmm