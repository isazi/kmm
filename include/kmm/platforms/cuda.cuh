#pragma once

#include "kmm/memory.hpp"
#include "kmm/types.hpp"

namespace kmm {

class CudaAllocation: public MemoryAllocation {
  public:
    explicit CudaAllocation(size_t nbytes);
    ~CudaAllocation();

    void* data() const {
        return m_data;
    }

    size_t size() const {
        return m_nbytes;
    }

  private:
    size_t m_nbytes;
    void* m_data;
};

class CudaMemory: public Memory {
  public:
    std::optional<std::unique_ptr<MemoryAllocation>> allocate(DeviceId id, size_t nbytes) override;

    void deallocate(DeviceId id, std::unique_ptr<MemoryAllocation> allocation) override;

    bool is_copy_possible(DeviceId src_id, DeviceId dst_id) override;

    void copy_async(
        DeviceId src_id,
        const MemoryAllocation* src_alloc,
        size_t src_offset,
        DeviceId dst_id,
        const MemoryAllocation* dst_alloc,
        size_t dst_offset,
        size_t nbytes,
        std::unique_ptr<MemoryCompletion> completion) override;

  private:
};

}  // namespace kmm
