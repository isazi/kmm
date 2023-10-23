#pragma once

#include <stdint.h>

namespace kmm {

using size_t = std::size_t;
using MemoryId = uint8_t;
using ExecutorId = uint8_t;
using BufferId = uint64_t;
using TaskId = uint64_t;

enum struct MemorySpace {
    Host,
    Cuda,
};

enum struct AccessMode { Read, Write };

class Allocation {
  public:
    virtual ~Allocation() = default;
};

class HostAllocation: Allocation {
  public:
    virtual void* as_void_ptr() const = 0;
};

struct BufferLayout {
    size_t size_in_bytes;
    size_t alignment;
};

struct BufferRequirement {
    BufferId buffer_id;
    MemoryId memory_id;
    AccessMode mode;
};

class Memory {
  public:
    virtual ~Memory() = default;
    virtual std::string name() const = 0;
    virtual std::optional<ExecutorId> executor_affinity() const {
        return {};
    }

    virtual std::optional<std::shared_ptr<Allocation>>
    allocate_buffer(BufferLayout layout) const = 0;
    virtual void release_buffer(std::shared_ptr<Allocation> alloc) const = 0;
};

}  // namespace kmm