#pragma once

#ifdef KMM_USE_CUDA
    #include <vector_types.h>
#endif

#include "kmm/cuda/device.hpp"
#include "kmm/task_argument.hpp"

#ifdef KMM_USE_CUDA

namespace kmm {

struct Cuda {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Cuda;

    Cuda(int device = 0) : m_device(device) {}

    DeviceId find_device(Runtime& rt) const {
        for (size_t i = 0, n = rt.num_devices(); i < n; i++) {
            auto id = DeviceId(uint8_t(i));

            if (const auto* info = dynamic_cast<const CudaDeviceInfo*>(&rt.device_info(id))) {
                if (info->device() == m_device) {
                    return id;
                }
            }
        }

        throw std::runtime_error("could not find cuda device");
    }

    template<typename F, typename... Args>
    void operator()(kmm::Device& device, kmm::TaskContext&, F&& fun, Args&&... args) const {
        std::forward<F>(fun)(device.cast<CudaDevice>(), std::forward<Args>(args)...);
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

    DeviceId find_device(Runtime& rt) const {
        return m_device.find_device(rt);
    }

    template<typename... KernelArgs, typename... Args>
    void operator()(
        kmm::Device& device,
        kmm::TaskContext&,
        void (*const kernel_function)(KernelArgs...),
        Args&&... args) const {
        device.cast<CudaDevice>().launch(
            {m_grid_dim.x, m_grid_dim.y, m_grid_dim.z},
            {m_block_dim.x, m_block_dim.y, m_block_dim.z},
            m_shared_memory,
            kernel_function,
            KernelArgs(std::forward<Args>(args))...);
    }

  private:
    Cuda m_device;
    dim3 m_grid_dim;
    dim3 m_block_dim;
    unsigned int m_shared_memory = 0;
};

}  // namespace kmm

#endif  // KMM_USE_CUDA