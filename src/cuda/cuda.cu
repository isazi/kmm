#include "spdlog/spdlog.h"

#include "kmm/cuda/cuda.cuh"

namespace kmm {

CudaAllocation::CudaAllocation(size_t nbytes) : m_nbytes(nbytes) {
    cudaError_t err = cudaMalloc(&m_data, m_nbytes);
    // TODO: actually handle the error
    if (err != cudaSuccess) {
        spdlog::error("Impossible to allocate CUDA memory: {}", int(err));
    }
}

CudaAllocation::~CudaAllocation() {
    cudaError_t err = cudaFree(m_data);
    // TODO: actually handle the error
    if (err != cudaSuccess) {
        spdlog::error("Impossible to free CUDA memory: {}", int(err));
    }
}

std::optional<std::unique_ptr<MemoryAllocation>> CudaMemory::allocate(MemoryId id, size_t nbytes) {
    return std::make_unique<CudaAllocation>(nbytes);
}

void CudaMemory::deallocate(MemoryId id, std::unique_ptr<MemoryAllocation> allocation) {
    // TODO: destroy the CudaAllocation
    auto& alloc = dynamic_cast<CudaAllocation&>(*allocation);
}

bool CudaMemory::is_copy_possible(MemoryId src_id, MemoryId dst_id) {
    if (src_id == 0 && dst_id == 0) {
        return false;
    }
    return true;
}

void CudaMemory::copy_async(
    MemoryId src_id,
    const MemoryAllocation* src_alloc,
    size_t src_offset,
    MemoryId dst_id,
    const MemoryAllocation* dst_alloc,
    size_t dst_offset,
    size_t num_bytes,
    MemoryCompletion completion) {
    completion.complete(ErrorPtr("not implemented"));
}

}  // namespace kmm