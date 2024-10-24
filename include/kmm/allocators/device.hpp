#pragma once

#include "base.hpp"

namespace kmm {

class PinnedMemoryAllocator: public SyncAllocator {
  public:
    PinnedMemoryAllocator(
        CudaContextHandle context,
        std::shared_ptr<CudaStreamManager> streams,
        size_t max_bytes = std::numeric_limits<size_t>::max()
    );

    bool allocate(size_t nbytes, void** addr_out) final;
    void deallocate(void* addr, size_t nbytes) final;

  private:
    CudaContextHandle m_context;
};

class DeviceMemoryAllocator: public SyncAllocator {
  public:
    DeviceMemoryAllocator(
        CudaContextHandle context,
        std::shared_ptr<CudaStreamManager> streams,
        size_t max_bytes = std::numeric_limits<size_t>::max()
    );

    bool allocate(size_t nbytes, void** addr_out) final;
    void deallocate(void* addr, size_t nbytes) final;

  private:
    CudaContextHandle m_context;
};

enum struct DevicePoolKind { Default, Create };

class DevicePoolAllocator: public AsyncAllocator {
  public:
    DevicePoolAllocator(
        CudaContextHandle context,
        std::shared_ptr<CudaStreamManager> streams,
        DevicePoolKind kind = DevicePoolKind::Create,
        size_t max_bytes = std::numeric_limits<size_t>::max()
    );
    ~DevicePoolAllocator();
    bool allocate_async(size_t nbytes, void** addr_out, DeviceEventSet* deps_out) final;
    void deallocate_async(void* addr, size_t nbytes, DeviceEventSet deps) final;
    void make_progress() final;
    void trim(size_t nbytes_remaining = 0) final;

  private:
    struct Allocation;

    CudaContextHandle m_context;
    CUmemoryPool m_pool;
    std::shared_ptr<CudaStreamManager> m_streams;
    DeviceStream m_alloc_stream;
    DeviceStream m_dealloc_stream;
    std::deque<Allocation> m_pending_deallocs;
    DevicePoolKind m_kind;
    size_t m_bytes_in_use = 0;
    size_t m_bytes_limit;
};

}  // namespace kmm