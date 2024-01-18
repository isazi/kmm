#pragma once

#include <memory>
#include <vector>

#include "fmt/format.h"

#include "kmm/block_header.hpp"
#include "kmm/event_list.hpp"
#include "kmm/identifiers.hpp"
#include "kmm/memory.hpp"

namespace kmm {

// Forward decl
class RuntimeImpl;
class Device;

/**
 * Represents an input for a task specifying the memory and block identifier.
 */
struct TaskInput {
    BlockId block_id;
    std::optional<MemoryId> memory_id;
};

/**
 * Represents an output of a task, containing a memory identifier and a block header.
 */
struct TaskOutput {
    std::unique_ptr<BlockHeader> header;
    MemoryId memory_id;
};

/**
 * Encapsulates the requirements for a task, including the device identifier and lists of inputs and outputs.
 */
struct TaskRequirements {
    TaskRequirements(DeviceId id) : device_id(id) {}

    size_t add_input(BlockId block_id);
    size_t add_input(BlockId block_id, MemoryId memory_id);
    size_t add_input(BlockId block_id, RuntimeImpl& rt);

    size_t add_output(std::unique_ptr<BlockHeader> header, MemoryId memory_id);
    size_t add_output(std::unique_ptr<BlockHeader> header, RuntimeImpl& rt);

    DeviceId device_id;
    EventList dependencies;
    std::vector<TaskInput> inputs = {};
    std::vector<TaskOutput> outputs = {};
};

/**
 * Provides read-only access to a block.
 */
struct BlockAccessor {
    BlockId block_id;
    std::shared_ptr<const BlockHeader> header;
    const MemoryAllocation* allocation = nullptr;
};

/**
 * Provides read-write access to a block.
 */
struct BlockAccessorMut {
    BlockId block_id;
    BlockHeader* header;
    MemoryAllocation* allocation = nullptr;
};

/**
 * Provides the context required to run task, such as the input and output data.
 */
struct TaskContext {
    TaskContext() = default;

    std::vector<BlockAccessor> inputs;
    std::vector<BlockAccessorMut> outputs;
};

/**
 * Abstract class representing a task to be executed.
 */
class Task {
  public:
    virtual ~Task() = default;
    virtual void execute(Device&, TaskContext&) = 0;
};

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