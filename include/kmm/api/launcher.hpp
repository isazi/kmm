#pragma once

#include <tuple>

#include "task_data.hpp"

namespace kmm {

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
    EventId operator()(const RuntimeImpl* runtime, Partition<N> dist, F fun, Args&&... args) {
        return execute_impl(
            std::index_sequence_for<Args...>(),
            runtime,
            dist,
            fun,
            std::forward<Args>(args)...);
    }

    template<size_t N, typename F, typename... Args, size_t... Is>
    static EventId execute_impl(
        std::index_sequence<Is...>,
        const RuntimeImpl* runtime,
        Partition<N> dist,
        F fun,
        Args&&... args) {
        TaskRequirements reqs;

        auto processors = std::make_tuple(TaskDataProcessor<std::decay_t<Args>> {args}...);
        auto events = EventList {};

        for (const auto& chunk : dist.chunks) {
            auto task =
                std::make_shared<HostTaskImpl<N, F, serialized_argument_t<std::decay_t<Args>>...>>(
                    chunk.local_size,
                    fun,
                    std::get<Is>(processors).pre_enqueue(chunk, reqs)...);

            auto result = runtime->enqueue_host_task(task, reqs);

            ((std::get<Is>(processors).post_enqueue(chunk, result)), ...);

            events.push_back(result.event_id);
        }

        ((std::get<Is>(processors).finalize()), ...);

        return runtime->join_events(events);
    }
};

}  // namespace kmm