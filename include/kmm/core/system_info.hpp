#pragma once

#include "identifiers.hpp"

#include "kmm/utils/cuda.hpp"

namespace kmm {

class CudaDeviceInfo {
  public:
    static constexpr size_t NUM_ATTRIBUTES = CU_DEVICE_ATTRIBUTE_MAX;

    CudaDeviceInfo(DeviceId id, CudaContextHandle context);

    /**
     * Returns the name of the CUDA device as provided by `cuDeviceGetName`.
     */
    std::string name() const {
        return m_name;
    }

    /**
     * Returns which memory this device has affinity to.
     */
    MemoryId memory_id() const {
        return MemoryId(m_id);
    }

    /**
     * Return this device as a `DeviceId`.
     */
    DeviceId device_id() const {
        return m_id;
    }

    /**
     * Return this device as a `CUdevice`.
     */
    CUdevice device_ordinal() const {
        return m_device_id;
    }

    /**
     * Returns the total memory size of this device.
     */
    size_t total_memory_size() const {
        return m_memory_capacity;
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
     * Returns the value of the provided attribute.
     */
    int attribute(CUdevice_attribute attrib) const;

  private:
    DeviceId m_id;
    std::string m_name;
    CUdevice m_device_id;
    size_t m_memory_capacity;
    std::array<int, NUM_ATTRIBUTES> m_attributes;
};

class SystemInfo {
  public:
    SystemInfo(std::vector<CudaDeviceInfo> devices = {});
    size_t num_devices() const;
    const CudaDeviceInfo& device(DeviceId id) const;
    const CudaDeviceInfo& device_by_ordinal(CUdevice ordinal) const;

  private:
    std::vector<CudaDeviceInfo> m_devices;
};

}  // namespace kmm