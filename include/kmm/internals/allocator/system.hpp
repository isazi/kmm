#pragma once

#include "base.hpp"

namespace kmm {

class SystemAllocator: public DirectMemoryAllocator {
  public:
    SystemAllocator(std::shared_ptr<CudaStreamManager> streams, size_t max_bytes=std::numeric_limits<size_t>::max()):
        DirectMemoryAllocator(streams, max_bytes) {}

  protected:
    bool allocate_impl(size_t nbytes, void*& addr_out) final;
    void deallocate_impl(void* addr_out, size_t nbytes) final;
};

}