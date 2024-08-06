#pragma once

#include <cuda.h>

#include "buffer.hpp"

namespace kmm {

struct TaskContext {
    std::vector<BufferAccessor> accessors;
};

class Task {
  public:
    virtual ~Task() = default;
};

class HostTask: public Task {
  public:
    virtual void execute(TaskContext& context) = 0;
};

class DeviceTask: public Task {
  public:
    virtual void submit(CUstream stream, TaskContext& context) = 0;
};

}  // namespace kmm