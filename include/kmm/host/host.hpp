#pragma once

#include "kmm/host/device.hpp"

namespace kmm {

struct Host {
    static constexpr ExecutionSpace execution_space = ExecutionSpace::Host;

    DeviceId find_device(Runtime& rt) const {
        for (size_t i = 0, n = rt.num_devices(); i < n; i++) {
            auto id = DeviceId(checked_cast<uint8_t>(i));

            if (dynamic_cast<const HostDeviceInfo*>(&rt.device_info(id)) != nullptr) {
                return id;
            }
        }

        throw std::runtime_error("could not find host device");
    }

    template<typename F, typename... Args>
    void operator()(Device&, TaskContext&, F&& fun, Args&&... args) const {
        std::forward<F>(fun)(std::forward<Args>(args)...);
    }
};

}  // namespace kmm