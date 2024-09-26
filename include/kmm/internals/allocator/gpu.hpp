#pragma once

#include "base.hpp"

namespace kmm {

struct Allocation {
    void* addr;
    size_t nbytes;
    GPUEvent event;
};

class PinnedMemoryAllocator: public DirectMemoryAllocator {
  public:
    PinnedMemoryAllocator(
        GPUContextHandle context,
        std::shared_ptr<GPUStreamManager> streams,
        size_t max_bytes=std::numeric_limits<size_t>::max());

  protected:
    bool allocate_impl(size_t nbytes, void*& addr_out) final;
    void deallocate_impl(void* addr, size_t nbytes) final;

  private:
    GPUContextHandle m_context;
};

class DeviceMemoryAllocator: public DirectMemoryAllocator {
  public:
    DeviceMemoryAllocator(
        GPUContextHandle context,
        std::shared_ptr<GPUStreamManager> streams,
        size_t max_bytes=std::numeric_limits<size_t>::max());

  protected:
    bool allocate_impl(size_t nbytes, void*& addr_out) final;
    void deallocate_impl(void* addr, size_t nbytes) final;

  private:
    GPUContextHandle m_context;
};


class DevicePoolAllocator: public MemoryAllocator {
  public:
    DevicePoolAllocator(GPUContextHandle context, std::shared_ptr<GPUStreamManager> streams, size_t max_bytes=std::numeric_limits<size_t>::max());
    ~DevicePoolAllocator();
    bool allocate(size_t nbytes, void*& addr_out, GPUEventSet& deps_out) final;
    void deallocate(void* addr, size_t nbytes, GPUEventSet deps) final;

  private:
    GPUContextHandle m_context;
    GPUmemoryPool m_pool;
    std::shared_ptr<GPUStreamManager> m_streams;
    GPUStream m_alloc_stream;
    GPUStream m_dealloc_stream;
    std::deque<Allocation> m_pending_deallocs;
    size_t m_bytes_in_use = 0;
    size_t m_bytes_limit;
};

}