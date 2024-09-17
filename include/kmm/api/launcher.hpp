#pragma once

#include "kmm/api/task_data.hpp"

namespace kmm {

template<typename F>
struct Host {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Host;

    Host(F fun) : m_fun(fun) {}

    template<size_t N>
    ProcessorId find_processor(const SystemInfo& info, Chunk<N> chunk) {
        return ProcessorId::host();
    }

    template<size_t N, typename... Args>
    void operator()(ExecutionContext& exec, Chunk<N> chunk, Args... args) {
        auto region = rect<N>(chunk.offset, chunk.size);
        m_fun(region, args...);
    }

  private:
    std::decay_t<F> m_fun;
};

template<typename F>
struct Cuda {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Cuda;

    Cuda(F fun) : m_fun(fun) {}

    template<size_t N>
    ProcessorId find_processor(const SystemInfo& info, Chunk<N> chunk) {
        return chunk.owner_id.is_device() ? chunk.owner_id : ProcessorId(DeviceId(0));
    }

    template<size_t N, typename... Args>
    void operator()(ExecutionContext& exec, Chunk<N> chunk, Args... args) {
        auto region = rect<N>(chunk.offset, chunk.size);
        m_fun(exec.cast<CudaDevice>(), region, args...);
    }

  private:
    std::decay_t<F> m_fun;
};

template<typename F>
struct CudaKernel {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Cuda;

    dim3 block_size;
    dim3 elements_per_block;
    uint32_t shared_memory;

    CudaKernel(F kernel, dim3 block_size) : CudaKernel(kernel, block_size, block_size) {}

    CudaKernel(F kernel, dim3 block_size, dim3 elements_per_block, uint32_t shared_memory = 0) :
        kernel(kernel),
        block_size(block_size),
        elements_per_block(elements_per_block),
        shared_memory(shared_memory) {}

    template<size_t N>
    ProcessorId find_processor(const SystemInfo& info, Chunk<N> chunk) {
        return chunk.owner_id.is_device() ? chunk.owner_id : DeviceId(0);
    }

    template<size_t N, typename... Args>
    void operator()(ExecutionContext& exec, Chunk<N> chunk, Args... args) {
        int64_t g[3] = {chunk.size.get(0), chunk.size.get(1), chunk.size.get(2)};
        int64_t b[3] = {elements_per_block.x, elements_per_block.y, elements_per_block.z};
        dim3 grid_dim = {
            checked_cast<unsigned int>((g[0] / b[0]) + int64_t(g[0] % b[0] != 0)),
            checked_cast<unsigned int>((g[1] / b[1]) + int64_t(g[1] % b[1] != 0)),
            checked_cast<unsigned int>((g[2] / b[2]) + int64_t(g[2] % b[2] != 0)),
        };

        auto region = rect<N>(chunk.offset, chunk.size);
        exec.cast<CudaDevice>().launch(  //
            grid_dim,
            block_size,
            shared_memory,
            kernel,
            region,
            args...);
    }

  private:
    std::decay_t<F> kernel;
};

template<size_t N, typename Launcher, typename... Args>
class TaskImpl: public Task {
  public:
    TaskImpl(Chunk<N> chunk, Launcher launcher, Args... args) :
        m_chunk(chunk),
        m_launcher(std::move(launcher)),
        m_args(std::move(args)...) {}

    void execute(ExecutionContext& device, TaskContext context) override {
        execute_impl(std::index_sequence_for<Args...>(), device, context);
    }

    template<size_t... Is>
    void execute_impl(std::index_sequence<Is...>, ExecutionContext& device, TaskContext& context) {
        static constexpr ExecutionSpace execution_space = Launcher::execution_space;

        m_launcher(
            device,
            m_chunk,
            TaskDataDeserialize<execution_space, Args>::unpack(context, std::get<Is>(m_args))...);
    }

  private:
    Chunk<N> m_chunk;
    Launcher m_launcher;
    std::tuple<Args...> m_args;
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
            TaskDataProcessor<typename std::decay<Args>::type> {std::forward<Args>(args)}...};

        for (const Chunk<N>& chunk : partition.chunks) {
            ProcessorId processor_id = launcher.find_processor(worker.system_info(), chunk);

            TaskBuilder builder {
                .graph = graph,
                .worker = worker,
                .memory_id = worker.system_info().affinity_memory(processor_id),
                .buffers = {},
                .dependencies = {}};

            auto task = std::make_shared<TaskImpl<
                N,
                Launcher,
                typename TaskDataProcessor<typename std::decay<Args>::type>::type...>>(
                chunk,
                launcher,
                std::get<Is>(processors).process_chunk(chunk, builder)...);

            EventId id = graph.insert_task(
                processor_id,
                std::move(task),
                std::move(builder.buffers),
                std::move(builder.dependencies));

            events.push_back(id);
        }

        TaskResult result =
            TaskResult {.graph = graph, .worker = worker, .events = std::move(events)};
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