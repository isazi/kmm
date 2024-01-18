#pragma once

#include <memory>
#include <thread>

#include "kmm/device.hpp"
#include "kmm/host/thread_pool.hpp"
#include "kmm/task_serialize.hpp"

namespace kmm {

class HostDeviceInfo final: public DeviceInfo {
    std::string name() const override;
    MemoryId memory_affinity() const override;
};

class ParallelDevice final: public Device {};

class ParallelDeviceHandle: public DeviceHandle, public ThreadPool {
  public:
    std::unique_ptr<DeviceInfo> info() const override;
    void submit(std::shared_ptr<Task>, TaskContext, Completion) const override;
};

}  // namespace kmm