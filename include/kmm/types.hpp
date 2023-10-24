#pragma once

#include <stdint.h>

namespace kmm {

using index_t = int;
using size_t = std::size_t;
using MemoryId = uint8_t;
using ExecutorId = uint8_t;
using BufferId = uint64_t;
using TaskId = uint64_t;

static constexpr BufferId INVALID_BUFFER_ID = ~uint64_t(0);
static constexpr TaskId INVALID_TASK_ID = ~uint64_t(0);
static constexpr MemoryId INVALID_MEMORY_ID = uint8_t(~0u);

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
    virtual void* data_as_ptr() const = 0;
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