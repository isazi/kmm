#pragma once

#include "kmm/api/task_data.hpp"

namespace kmm {

struct Host {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Host;

    template<size_t N, typename F, typename... Args>
    void operator()(ExecutionContext& exec, Chunk<N> chunk, F fun, Args... args) {
        fun(args...);
    }
};

struct Cuda {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Cuda;

    template<size_t N, typename F, typename... Args>
    void operator()(ExecutionContext& exec, Chunk<N> chunk, F fun, Args... args) {
        fun(exec.cast<CudaDevice>(), args...);
    }
};

struct CudaKernel {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Cuda;

    dim3 block_size;
    dim3 elements_per_block;
    uint32_t shared_memory;

    CudaKernel(dim3 block_size) : CudaKernel(block_size, block_size) {}

    CudaKernel(dim3 block_size, dim3 elements_per_block, uint32_t shared_memory = 0) :
        block_size(block_size),
        elements_per_block(elements_per_block),
        shared_memory(shared_memory) {}

    template<size_t N, typename K, typename... Args>
    void operator()(ExecutionContext& exec, Chunk<N> chunk, K kernel, Args... args) {
        dim3 grid_dim = {
            checked_cast<unsigned int>(chunk.size.get(0) / long(elements_per_block.x) + 1),
            checked_cast<unsigned int>(chunk.size.get(1) / long(elements_per_block.y) + 1),
            checked_cast<unsigned int>(chunk.size.get(2) / long(elements_per_block.z) + 1),
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
            std::forward<Args>(args)...};

        for (const Chunk<N>& chunk : partition.chunks) {
            TaskBuilder builder {
                .graph = graph,
                .worker = worker,
                .memory_id = chunk.owner_id.as_memory(),
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
                chunk.owner_id,
                std::move(task),
                std::move(builder.buffers),
                std::move(builder.dependencies));

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