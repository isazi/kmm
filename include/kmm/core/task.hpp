#pragma once

#include <cuda.h>

#include "buffer.hpp"

namespace kmm {

struct TaskContext {
    std::vector<BufferAccessor> accessors;
};

class HostTask {
  public:
    virtual ~HostTask() = default;
    virtual void execute(TaskContext& context) = 0;
};

class DeviceTask {
  public:
    virtual ~DeviceTask() = default;
    virtual void execute(TaskContext& context) = 0;
};

}  // namespace kmm