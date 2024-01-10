#pragma once

#include <vector_types.h>

#include "kmm/cuda/executor.hpp"
#include "kmm/task_serialize.hpp"

namespace kmm {

struct CudaLauncher {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Host;

    template<typename F, typename... Args>
    void operator()(kmm::Executor& executor, kmm::TaskContext&, F&& fun, Args&&... args) const {
        std::forward<F>(fun)(executor.cast<CudaExecutor>(), std::forward<Args>(args)...);
    }
};

struct Cuda {
    Cuda(int device = 0) : m_device(device) {}

    ExecutorId find_executor(RuntimeImpl& rt) const {
        for (size_t i = 0, n = rt.num_executors(); i < n; i++) {
            auto id = ExecutorId(uint8_t(i));
            auto* info = dynamic_cast<const CudaExecutorInfo*>(&rt.executor_info(id));

            if (info != nullptr && info->device() == m_device) {
                return id;
            }
        }

        throw std::runtime_error("could not find cuda executor");
    }

    template<typename Fun, typename... Args>
    EventId operator()(RuntimeImpl& rt, Fun&& fun, Args&&... args) const {
        return TaskLauncher<CudaLauncher, Fun, Args...>::call(
            CudaLauncher {},
            find_executor(rt),
            rt,
            fun,
            args...);
    }

  private:
    int m_device;
};

struct CudaKernelLauncher {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Host;

    CudaKernelLauncher(dim3 grid_dim, dim3 block_dim, unsigned int shared_memory = 0) :
        m_grid_dim(grid_dim),
        m_block_dim(block_dim),
        m_shared_memory(shared_memory) {}

    template<typename F, typename... Args>
    void operator()(kmm::Executor& executor, kmm::TaskContext&, F kernel, Args... args) const {
        executor.cast<CudaExecutor>().launch(
            {m_grid_dim.x, m_grid_dim.y, m_grid_dim.z},
            {m_block_dim.x, m_block_dim.y, m_block_dim.z},
            m_shared_memory,
            kernel,
            args...);
    }

  private:
    dim3 m_grid_dim;
    dim3 m_block_dim;
    unsigned int m_shared_memory = 0;
};

struct CudaKernel {
    CudaKernel(dim3 grid_dim, dim3 block_dim, Cuda device = {}) :
        m_device(device),
        m_launcher(grid_dim, block_dim) {}

    template<typename KernelFun, typename... Args>
    EventId operator()(RuntimeImpl& rt, KernelFun&& kernel_fun, Args&&... args) const {
        return TaskLauncher<CudaKernelLauncher, KernelFun, Args...>::call(
            m_launcher,
            m_device.find_executor(rt),
            rt,
            kernel_fun,
            args...);
    }

  private:
    Cuda m_device;
    CudaKernelLauncher m_launcher;
};

}  // namespace kmm
