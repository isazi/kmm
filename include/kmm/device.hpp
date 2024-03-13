#pragma once

#include <memory>
#include <vector>

#include "fmt/format.h"

#include "kmm/block_header.hpp"
#include "kmm/event_list.hpp"
#include "kmm/identifiers.hpp"
#include "kmm/memory.hpp"
#include "kmm/task.hpp"

namespace kmm {

/**
 * Exception throw if
 */
class InvalidDeviceException: public std::exception {
  public:
    InvalidDeviceException(const std::type_info& expected, const std::type_info& gotten);
    const char* what() const noexcept override;

  private:
    std::string m_message;
};

/**
 * Represents information of an device.
 */
class DeviceInfo {
  public:
    virtual ~DeviceInfo() = default;

    /**
     * The name of the device. Useful for debugging.
     */
    virtual std::string name() const = 0;

    /**
     * Which memory does this device has the strongest affinity to.
     */
    virtual MemoryId memory_affinity() const = 0;

    /**
     * Can the compute units from this device access the specified memory?
     */
    virtual bool is_memory_accessible(MemoryId id) const {
        return id == memory_affinity();
    }
};

/**
 * Abstract class that allows to submit tasks onto a device.
 */
class DeviceHandle {
  public:
    virtual ~DeviceHandle() = default;
    virtual std::unique_ptr<DeviceInfo> info() const = 0;
    virtual void submit(std::shared_ptr<Task>, TaskContext, Completion) const = 0;
};

/**
 * Represents the context in which an device operates.
 */
class Device {
  public:
    virtual ~Device() = default;

    template<typename T>
    T* cast_if() {
        return dynamic_cast<T*>(this);
    }

    template<typename T>
    const T* cast_if() const {
        return dynamic_cast<const T*>(this);
    }

    template<typename T>
    T& cast() {
        if (auto* ptr = this->template cast_if<T>()) {
            return *ptr;
        }

        throw InvalidDeviceException(typeid(T), typeid(*this));
    }

    template<typename T>
    const T& cast() const {
        if (auto* ptr = this->template cast_if<T>()) {
            return *ptr;
        }

        throw InvalidDeviceException(typeid(T), typeid(*this));
    }
};
}  // namespace kmm