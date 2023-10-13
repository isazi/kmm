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

}  // namespace kmm