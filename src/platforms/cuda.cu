
#include "spdlog/spdlog.h"

#include "kmm/platforms/cuda.cuh"

namespace kmm {

CudaAllocation::CudaAllocation(size_t nbytes) : m_nbytes(nbytes) {
    cudaError_t err = cudaMalloc(&m_data, m_nbytes);
    // TODO: actually handle the error
    if (err != cudaSuccess) {
        spdlog::error("Impossible to allocate CUDA memory: {}", err);
    }
}

CudaAllocation::~CudaAllocation() {
    cudaError_t err = cudaFree(m_data);
    // TODO: actually handle the error
    if (err != cudaSuccess) {
        spdlog::error("Impossible to free CUDA memory: {}", err);
    }
}

std::optional<std::unique_ptr<MemoryAllocation>> CudaMemory::allocate(DeviceId id, size_t nbytes) {
    return std::make_unique<CudaAllocation>(nbytes);
}

void CudaMemory::deallocate(DeviceId id, std::unique_ptr<MemoryAllocation> allocation) {
    // TODO: destroy the CudaAllocation
    auto& alloc = dynamic_cast<CudaAllocation&>(*allocation);
}

bool CudaMemory::is_copy_possible(DeviceId src_id, DeviceId dst_id) {
    if (src_id == 0 && dst_id == 0) {
        return false;
    }
    return true;
}

}  // namespace kmm