#pragma once
#include "kmm/internals/cuda_stream_manager.hpp"
#include "kmm/utils/macros.hpp"

namespace kmm {

class MemoryAllocator {
  public:
    virtual ~MemoryAllocator() = default;
    virtual bool allocate(size_t nbytes, void*& addr_out, CudaEventSet& deps_out) = 0;
    virtual void deallocate(void* addr, size_t nbytes, CudaEventSet deps = {}) = 0;
    virtual void make_progress() {}
};


class DirectMemoryAllocator: public MemoryAllocator {
  public:
    DirectMemoryAllocator(std::shared_ptr<CudaStreamManager> streams, size_t max_bytes=std::numeric_limits<size_t>::max());
    ~DirectMemoryAllocator();
    bool allocate(size_t nbytes, void*& addr_out, CudaEventSet& deps_out) final;
    void deallocate(void* addr, size_t nbytes, CudaEventSet deps) final;
    void make_progress() final;

  protected:
    virtual bool allocate_impl(size_t nbytes, void*& addr_out) = 0;
    virtual void deallocate_impl(void* addr, size_t nbytes) = 0;

  private:
    struct DeferredDealloc;

    std::shared_ptr<CudaStreamManager> m_streams;
    std::deque<DeferredDealloc> m_pending_deallocs;
    size_t m_bytes_in_use = 0;
    size_t m_bytes_limit;
};



}  // namespace kmm