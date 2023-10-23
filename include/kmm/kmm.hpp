#pragma once

#include <cstddef>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime.hpp"
#include "task.hpp"
#include "types.hpp"

namespace kmm {

// Compute devices
class DeviceType {
  public:
    DeviceType() = default;
    DeviceType(const DeviceType&) = delete;

  public:
    virtual ~DeviceType() = default;
};

class UnknownDevice: public DeviceType {};

class CPU: public DeviceType {};

class GPU: public DeviceType {
  public:
    GPU();
    GPU(unsigned int device_id);
    virtual ~GPU() = default;
    unsigned int device_id;
};

class CUDA: public GPU {
    using GPU::GPU;
};

// Stream

class Stream;
#ifdef USE_CUDA
    #include "kmm.cuh"
#endif

// Buffer

class Buffer {
  public:
    Buffer();
    Buffer(std::size_t size);
    Buffer(std::size_t size, CUDAPinned& memory);
    Buffer(CPU& device, std::size_t size);
    Buffer(CUDA& device, std::size_t size);
    // Manipulate size
    std::size_t size() const;
    void set_size(std::size_t size);
    // Manipulate device
    std::shared_ptr<DeviceType> device();
    void set_device(CPU& device);
    void set_device(CUDA& device);
    // Manipulate special memory
    std::shared_ptr<MemoryType> memory();
    void set_memory(CUDAPinned& memory);
    // Return true if the buffer is allocated
    bool is_allocated() const;
    // Allocate memory
    void allocate();
    void allocate(CUDAPinned& memory);
    void allocate(CUDA& device, Stream& stream);
    // Free memory
    void destroy();
    void destroy(CUDA& device, Stream& stream);
    // Return a pointer to the allocated buffer
    void* getPointer();

  private:
    void* buffer_;
    std::size_t size_;
    std::shared_ptr<DeviceType> device_;
    std::shared_ptr<MemoryType> memory_;
};

// Misc

inline bool is_cpu(DeviceType& device) {
    return dynamic_cast<const CPU*>(&device) != nullptr;
}

inline bool is_cuda_pinned(Buffer& buffer) {
    return dynamic_cast<CUDAPinned*>(buffer.memory().get()) != nullptr;
}

inline bool is_pinned(Buffer& buffer) {
    return is_cuda_pinned(buffer);
}

inline bool on_cpu(Buffer& buffer) {
    return dynamic_cast<CPU*>(buffer.device().get()) != nullptr;
}

inline bool on_cuda(Buffer& buffer) {
    return dynamic_cast<CUDA*>(buffer.device().get()) != nullptr;
}

inline bool same_device(const DeviceType& a, const DeviceType& b) {
    if (dynamic_cast<const CPU*>(&a) != nullptr) {
        return dynamic_cast<const CPU*>(&b) != nullptr;
    }

    if (auto a_ptr = dynamic_cast<const GPU*>(&a)) {
        if (auto b_ptr = dynamic_cast<const GPU*>(&b)) {
            return a_ptr->device_id == b_ptr->device_id;
        }
    }

    return false;
}

void cudaCopyD2H(CUDA& device, Buffer& source, Buffer& target, Stream& stream);
void cudaCopyD2H(CUDA& device, Buffer& source, void* target, Stream& stream);
void cudaCopyH2D(CUDA& device, Buffer& source, Buffer& target, Stream& stream);
void cudaCopyD2D(CUDA& device, Buffer& source, Buffer& target, Stream& stream);

// Manager

// Pointer

template<typename Type>
Pointer<Type>::Pointer() {}

template<typename Type>
Pointer<Type>::Pointer(BufferId id) {
    this->id_ = id;
}

// WritePointer

template<typename P>
class WritePointer {
  public:
    WritePointer(P& inner) : inner_(inner) {}

    P& get() const {
        return inner_;
    }

  private:
    P& inner_;
};

template<typename P>
WritePointer<P> write(P& ptr) {
    return ptr;
}

}  // namespace kmm
