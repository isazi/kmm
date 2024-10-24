#pragma once

#include "base.hpp"

namespace kmm {

class SystemAllocator: public SyncAllocator {
  public:
    SystemAllocator(
        std::shared_ptr<CudaStreamManager> streams,
        size_t max_bytes = std::numeric_limits<size_t>::max()
    ) :
        SyncAllocator(streams, max_bytes) {}

  protected:
    bool allocate(size_t nbytes, void** addr_out) final;
    void deallocate(void* addr, size_t nbytes) final;
};

}  // namespace kmm