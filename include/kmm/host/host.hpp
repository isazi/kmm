#pragma once
#include "executor.hpp"

#include "kmm/executor.hpp"

namespace kmm {

struct HostLauncher {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Host;

    template<typename F, typename... Args>
    void operator()(Executor&, TaskContext&, F&& fun, Args&&... args) const {
        std::forward<F>(fun)(std::forward<Args>(args)...);
    }
};

struct Host {
    template<typename Fun, typename... Args>
    EventId operator()(RuntimeImpl& rt, Fun&& fun, Args&&... args) const {
        return TaskLauncher<HostLauncher, Fun, Args...>::call(
            HostLauncher {},
            find_executor(rt),
            rt,
            fun,
            args...);
    }

    ExecutorId find_executor(RuntimeImpl& rt) const {
        for (size_t i = 0, n = rt.num_executors(); i < n; i++) {
            auto id = ExecutorId(i);

            if (dynamic_cast<const HostExecutorInfo*>(&rt.executor_info(id)) != nullptr) {
                return id;
            }
        }

        throw std::runtime_error("could not find host executor");
    }
};

}  // namespace kmm