#pragma once

#include <tuple>

#include "task_data.hpp"

namespace kmm {

template<size_t N, typename Launcher, typename F, typename... Args>
EventId submit_with_launcher(
    const RuntimeImpl* runtime,
    Partition<N> dist,
    Launcher launcher,
    F fun,
    Args&&... args) {
    return submit_with_launcher_impl(
        std::index_sequence_for<Args...>(),
        runtime,
        dist,
        launcher,
        fun,
        std::forward<Args>(args)...);
}

template<size_t N, typename Launcher, typename F, typename... Args, size_t... Is>
EventId submit_with_launcher_impl(
    std::index_sequence<Is...>,
    const RuntimeImpl* runtime,
    Partition<N> dist,
    Launcher launcher,
    F fun,
    Args&&... args) {
    auto processors = std::make_tuple(TaskDataProcessor<std::decay_t<Args>> {args}...);
    auto events = EventList {};

    for (const auto& chunk : dist.chunks) {
        TaskRequirements reqs;

        auto result = launcher(
            runtime,
            reqs,
            fun,
            chunk,
            std::get<Is>(processors).pre_enqueue(chunk, reqs)...);

        ((std::get<Is>(processors).post_enqueue(chunk, result)), ...);

        events.push_back(result.event_id);
    }

    ((std::get<Is>(processors).finalize()), ...);

    return runtime->join_events(events);
}

template<size_t N, typename F, typename... Args>
struct HostTaskImpl: public HostTask {
    HostTaskImpl(rect<N> local_size, F fun, Args... args) :
        m_local_size(local_size),
        m_fun(fun),
        m_args(std::move(args)...) {}

    void execute(TaskContext& context) override {
        execute_impl(std::index_sequence_for<Args...> {}, context);
    }

  private:
    template<size_t... Is>
    void execute_impl(std::index_sequence<Is...>, TaskContext& context) {
        m_fun(
            m_local_size,
            TaskDataDeserializer<ExecutionSpace::Host, Args> {}.deserialize(
                context,
                std::get<Is>(m_args))...);
    }

    rect<N> m_local_size;
    F m_fun;
    std::tuple<Args...> m_args;
};

struct Host {
    template<size_t N, typename F, typename... Args>
    TaskResult operator()(
        const RuntimeImpl* runtime,
        TaskRequirements& reqs,
        F fun,
        Chunk<N> chunk,
        Args&&... args) {
        auto task = std::make_shared<HostTaskImpl<N, F, Args...>>(chunk.local_size, fun, args...);
        return runtime->enqueue_host_task(task, reqs);
    }
};

struct CudaKernelLaunch {
    dim3 grid_size;
    dim3 block_size;
    unsigned int shared_memory;
};

struct CudaKernel {
    CudaKernel(dim3 block_size, dim3 grid_divisor, unsigned int shared_memory = 0) {}

    CudaKernel(dim3 block_size) : CudaKernel(block_size, block_size) {}

    template<size_t N, typename F, typename... Args>
    TaskResult operator()(
        const RuntimeImpl* runtime,
        TaskRequirements& reqs,
        F fun,
        Chunk<N> chunk,
        Args&&... args) {
        auto task = std::make_shared<HostTaskImpl<N, F, Args...>>(chunk.local_size, fun, args...);
        return runtime->enqueue_host_task(task, reqs);
    }
};

}  // namespace kmm