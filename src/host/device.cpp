#include <algorithm>
#include <utility>

#include "kmm/host/device.hpp"
#include "kmm/panic.hpp"

namespace kmm {

std::string HostDeviceInfo::name() const {
    return "CPU";
}

MemoryId HostDeviceInfo::memory_affinity() const {
    return MemoryId(0);
}

std::unique_ptr<DeviceInfo> ParallelDeviceHandle::info() const {
    return std::make_unique<HostDeviceInfo>();
}

void ParallelDeviceHandle::submit(
    std::shared_ptr<Task> task,
    TaskContext context,
    Completion completion) const {
    this->submit_task(  //
        std::move(task),
        std::move(context),
        std::move(completion));
}

}  // namespace kmm