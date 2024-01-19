#pragma once

#include "kmm/cuda/types.hpp"
#include "kmm/device.hpp"

#ifdef KMM_USE_CUDA

namespace kmm {

/**
 * Stores the information on a CUDA device.
 */
class CudaDeviceInfo: public DeviceInfo {
  public:
    static constexpr size_t NUM_ATTRIBUTES = CU_DEVICE_ATTRIBUTE_MAX;

    CudaDeviceInfo(CudaContextHandle context, MemoryId affinity_id);

    /**
     * Returns the name of the CUDA device as provided by `cuDeviceGetName`.
     */
    std::string name() const override {
        return m_name;
    }

    /**
     * Returns which memory this device has affinity to.
     */
    MemoryId memory_affinity() const override {
        return m_affinity_id;
    }

    /**
     * Return this device as a `CUdevice`.
     */
    CUdevice device() const {
        return m_device_id;
    }

    /**
     * Returns the maximum block size supported by this device.
     */
    dim3 max_block_dim() const;

    /**
     * Returns the maximum grid size supported by this device.
     */
    dim3 max_grid_dim() const;

    /**
     * Returns the compute capability of this device as integer `MAJOR * 10 + MINOR` (For example,
     * `86` means capability 8.6)
     */
    int compute_capability() const;

    /**
     * Returns the maximum number of threads per block supported by this device.
     */
    int max_threads_per_block() const;

    /**
     * Returns the total memory size of this device.
     */
    size_t total_memory_size() const;

    /**
     * Returns the value of the provided attribute.
     */
    int attribute(CUdevice_attribute attrib) const;

  private:
    std::string m_name;
    CUdevice m_device_id;
    size_t m_memory_capacity;
    MemoryId m_affinity_id;
    std::array<int, NUM_ATTRIBUTES> m_attributes;
};
}  // namespace kmm

#endif