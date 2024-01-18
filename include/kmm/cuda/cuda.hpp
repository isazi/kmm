#pragma once

#ifdef KMM_USE_CUDA
    #include <vector_types.h>
#endif

#include "kmm/cuda/executor.hpp"
#include "kmm/task_serialize.hpp"

#ifdef KMM_USE_CUDA

namespace kmm {

struct Cuda {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Cuda;

    Cuda(int device = 0) : m_device(device) {}

    ExecutorId find_executor(RuntimeImpl& rt) const {
        for (size_t i = 0, n = rt.num_executors(); i < n; i++) {
            auto id = ExecutorId(uint8_t(i));

            if (const auto* info = dynamic_cast<const CudaExecutorInfo*>(&rt.executor_info(id))) {
                if (info->device() == m_device) {
                    return id;
                }
            }
        }

        throw std::runtime_error("could not find cuda executor");
    }

    template<typename F, typename... Args>
    void operator()(kmm::Executor& executor, kmm::TaskContext&, F&& fun, Args&&... args) const {
        std::forward<F>(fun)(executor.cast<CudaExecutor>(), std::forward<Args>(args)...);
    }

  private:
    int m_device;
};

struct CudaKernel {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Cuda;

    CudaKernel(dim3 grid_dim, dim3 block_dim, Cuda device = {}) :
        m_device(device),
        m_grid_dim(grid_dim),
        m_block_dim(block_dim) {}

    ExecutorId find_executor(RuntimeImpl& rt) const {
        return m_device.find_executor(rt);
    }

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
    Cuda m_device;
    dim3 m_grid_dim;
    dim3 m_block_dim;
    unsigned int m_shared_memory = 0;
};

}  // namespace kmm

#endif  // KMM_USE_CUDA