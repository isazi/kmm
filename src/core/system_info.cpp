#include "fmt/format.h"

#include "kmm/core/system_info.hpp"

#ifdef KMM_USE_CUDA
namespace kmm {

CudaDeviceInfo::CudaDeviceInfo(DeviceId id, CudaContextHandle context) : m_id(id) {
    CudaContextGuard guard {context};

    KMM_CUDA_CHECK(cuCtxGetDevice(&m_device_id));

    char name[1024];
    KMM_CUDA_CHECK(cuDeviceGetName(name, 1024, m_device_id));
    m_name = std::string(name);

    for (size_t i = 1; i < NUM_ATTRIBUTES; i++) {
        auto attr = CUdevice_attribute(i);
        KMM_CUDA_CHECK(cuDeviceGetAttribute(&m_attributes[i], attr, m_device_id));
    }

    size_t ignore_free_memory;
    KMM_CUDA_CHECK(cuMemGetInfo(&ignore_free_memory, &m_memory_capacity));
}

dim3 CudaDeviceInfo::max_block_dim() const {
    return dim3(
        attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
        attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
        attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z));
}

dim3 CudaDeviceInfo::max_grid_dim() const {
    return dim3(
        attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X),
        attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y),
        attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z));
}

int CudaDeviceInfo::compute_capability() const {
    return attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) * 10
        + attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
}

int CudaDeviceInfo::max_threads_per_block() const {
    return attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
}

int CudaDeviceInfo::attribute(CUdevice_attribute attrib) const {
    if (attrib < NUM_ATTRIBUTES) {
        return m_attributes[attrib];
    }

    throw std::runtime_error("unsupported attribute requested");
}

SystemInfo::SystemInfo(std::vector<CudaDeviceInfo> devices) : m_devices(devices) {}

size_t SystemInfo::num_devices() const {
    return m_devices.size();
}

const CudaDeviceInfo& SystemInfo::device(DeviceId id) const {
    return m_devices.at(id.get());
}

const CudaDeviceInfo& SystemInfo::device_by_ordinal(CUdevice ordinal) const {
    for (auto& device : m_devices) {
        if (device.device_ordinal() == ordinal) {
            return device;
        }
    }

    throw std::runtime_error(fmt::format("cannot find CUDA device with ordinal {}", ordinal));
}

}  // namespace kmm
#endif
