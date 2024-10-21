#pragma once

#include "base.hpp"

namespace kmm {

class PinnedMemoryAllocator: public DirectMemoryAllocator {
  public:
    PinnedMemoryAllocator(
        CudaContextHandle context,
        std::shared_ptr<CudaStreamManager> streams,
        size_t max_bytes=std::numeric_limits<size_t>::max());

  protected:
    bool allocate_impl(size_t nbytes, void*& addr_out) final;
    void deallocate_impl(void* addr, size_t nbytes) final;

  private:
    CudaContextHandle m_context;
};

class DeviceMemoryAllocator: public DirectMemoryAllocator {
  public:
    DeviceMemoryAllocator(
        CudaContextHandle context,
        std::shared_ptr<CudaStreamManager> streams,
        size_t max_bytes=std::numeric_limits<size_t>::max());

  protected:
    bool allocate_impl(size_t nbytes, void*& addr_out) final;
    void deallocate_impl(void* addr, size_t nbytes) final;

  private:
    CudaContextHandle m_context;
};


class DevicePoolAllocator: public MemoryAllocator {
  public:
    DevicePoolAllocator(CudaContextHandle context, std::shared_ptr<CudaStreamManager> streams, size_t max_bytes=std::numeric_limits<size_t>::max());
    ~DevicePoolAllocator();
    bool allocate(size_t nbytes, void*& addr_out, DeviceEventSet& deps_out) final;
    void deallocate(void* addr, size_t nbytes, DeviceEventSet deps) final;

  private:
    struct Allocation;

    CudaContextHandle m_context;
    CUmemoryPool m_pool;
    std::shared_ptr<CudaStreamManager> m_streams;
    CudaStream m_alloc_stream;
    CudaStream m_dealloc_stream;
    std::deque<Allocation> m_pending_deallocs;
    size_t m_bytes_in_use = 0;
    size_t m_bytes_limit;
};

}