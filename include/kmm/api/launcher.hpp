#pragma once

#include "kmm/api/argument.hpp"
#include "kmm/core/execution_context.hpp"
#include "kmm/worker/worker.hpp"

namespace kmm {

template<typename F>
struct Host {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Host;

    Host(F fun) : m_fun(fun) {}

    template<typename... Args>
    void operator()(ExecutionContext& exec, WorkChunk chunk, Args... args) {
        auto region = NDRange(chunk.offset, chunk.size);
        m_fun(region, args...);
    }

  private:
    std::decay_t<F> m_fun;
};

template<typename F>
struct GPU {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Device;

    GPU(F fun) : m_fun(fun) {}

    template<typename... Args>
    void operator()(ExecutionContext& exec, WorkChunk chunk, Args... args) {
        auto region = NDRange(chunk.offset, chunk.size);
        m_fun(exec.cast<DeviceContext>(), region, args...);
    }

  private:
    std::decay_t<F> m_fun;
};

template<typename F>
struct GPUKernel {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Device;

    GPUKernel(F kernel, dim3 block_size) : GPUKernel(kernel, block_size, block_size) {}

    GPUKernel(F kernel, dim3 block_size, dim3 elements_per_block, uint32_t shared_memory = 0) :
        kernel(kernel),
        block_size(block_size),
        elements_per_block(elements_per_block),
        shared_memory(shared_memory) {}

    template<typename... Args>
    void operator()(ExecutionContext& exec, WorkChunk chunk, Args... args) {
        int64_t g[3] = {chunk.size.get(0), chunk.size.get(1), chunk.size.get(2)};
        int64_t b[3] = {elements_per_block.x, elements_per_block.y, elements_per_block.z};
        dim3 grid_dim = {
            checked_cast<unsigned int>((g[0] / b[0]) + int64_t(g[0] % b[0] != 0)),
            checked_cast<unsigned int>((g[1] / b[1]) + int64_t(g[1] % b[1] != 0)),
            checked_cast<unsigned int>((g[2] / b[2]) + int64_t(g[2] % b[2] != 0)),
        };

        auto region = NDRange(chunk.offset, chunk.size);
        exec.cast<DeviceContext>().launch(  //
            grid_dim,
            block_size,
            shared_memory,
            kernel,
            region,
            args...
        );
    }

  private:
    std::decay_t<F> kernel;
    dim3 block_size;
    dim3 elements_per_block;
    uint32_t shared_memory;
};

template<typename Launcher, typename... Args>
class TaskImpl: public Task {
  public:
    TaskImpl(WorkChunk chunk, Launcher launcher, Args... args) :
        m_chunk(chunk),
        m_launcher(std::move(launcher)),
        m_args(std::move(args)...) {}

    void execute(ExecutionContext& exec, TaskContext context) override {
        execute_impl(std::index_sequence_for<Args...>(), exec, context);
    }

    template<size_t... Is>
    void execute_impl(std::index_sequence<Is...>, ExecutionContext& exec, TaskContext& context) {
        static constexpr ExecutionSpace execution_space = Launcher::execution_space;

        m_launcher(
            exec,
            m_chunk,
            ArgumentUnpack<execution_space, Args>::unpack(context, std::get<Is>(m_args))...
        );
    }

  private:
    WorkChunk m_chunk;
    Launcher m_launcher;
    std::tuple<Args...> m_args;
};

namespace detail {
template<size_t... Is, typename Launcher, typename... Args>
EventId parallel_submit_impl(
    std::index_sequence<Is...>,
    Worker& worker,
    const SystemInfo& system_info,
    const WorkPartition& partition,
    Launcher launcher,
    Args&&... args
) {
    std::tuple<ArgumentHandler<std::decay_t<Args>>...> handlers = {
        ArgumentHandler<std::decay_t<Args>> {std::forward<Args>(args)}...};

    return worker.with_task_graph([&](TaskGraph& graph) {
        EventList events;

        auto init = TaskGroupInfo {
            .worker = worker,  //
            .graph = graph,
            .partition = partition};

        (std::get<Is>(handlers).initialize(init), ...);

        for (const WorkChunk& chunk : partition.chunks) {
            ProcessorId processor_id = chunk.owner_id;

            auto instance = TaskInstance {
                .worker = worker,
                .graph = graph,
                .chunk = chunk,
                .memory_id = system_info.affinity_memory(processor_id),
                .buffers = {},
                .dependencies = {}};

            auto task = std::make_shared<TaskImpl<Launcher, packed_argument_t<Args>...>>(
                chunk,
                launcher,
                std::get<Is>(handlers).process_chunk(instance)...
            );

            EventId event_id = graph.insert_task(
                processor_id,
                std::move(task),
                std::move(instance.buffers),
                std::move(instance.dependencies)
            );

            events.push_back(event_id);
        }

        auto result = TaskGroupResult {
            .worker = worker,  //
            .graph = graph,
            .events = std::move(events)};

        (std::get<Is>(handlers).finalize(result), ...);

        return graph.join_events(result.events);
    });
}
}  // namespace detail

template<typename Launcher, typename... Args>
EventId parallel_submit(
    Worker& worker,
    const SystemInfo& system_info,
    const WorkPartition& partition,
    Launcher launcher,
    Args&&... args
) {
    return detail::parallel_submit_impl(
        std::index_sequence_for<Args...> {},
        worker,
        system_info,
        partition,
        launcher,
        std::forward<Args>(args)...
    );
}

}  // namespace kmm
