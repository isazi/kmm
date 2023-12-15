#pragma once

#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>

#include "kmm/types.hpp"

namespace kmm {

class MemoryAllocation {
  public:
    virtual ~MemoryAllocation() = default;
};

class TransferCompletion {
  public:
    virtual ~TransferCompletion() = default;
    virtual void complete() = 0;
};

class Memory {
  public:
    virtual ~Memory() = default;
    virtual std::optional<std::unique_ptr<MemoryAllocation>> allocate(
        kmm::DeviceId id,
        size_t num_bytes) = 0;
    virtual void deallocate(kmm::DeviceId id, std::unique_ptr<MemoryAllocation> allocation) = 0;

    virtual void copy_async(
        kmm::DeviceId src_id,
        const MemoryAllocation* src_alloc,
        size_t src_offset,
        kmm::DeviceId dst_id,
        const MemoryAllocation* dst_alloc,
        size_t dst_offset,
        size_t num_bytes,
        std::unique_ptr<TransferCompletion> completion) = 0;

    virtual bool is_copy_possible(kmm::DeviceId src_id, kmm::DeviceId dst_id) = 0;
};

}  // namespace kmm