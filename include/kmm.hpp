#include <cstddef>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#pragma once

namespace kmm {

// Data types

class DataType {
  public:
    virtual ~DataType() = default;
};

class UInteger: public DataType {};

class Integer: public DataType {};

class FP_Single: public DataType {};

class FP_Double: public DataType {};

// Compute devices

class DeviceType {
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

// Memories

class MemoryType {
  public:
    virtual ~MemoryType() = default;
};

class DefaultMemory: public MemoryType {};

class CUDAPinned: public DefaultMemory {};

// Pointer

template<typename Type>
class Pointer {
  public:
    Pointer();
    Pointer(unsigned int id);
    bool dirty;
    unsigned int id;
    Type type;
};

template<typename Type>
class WritePointer: public Pointer<Type> {
  public:
    WritePointer(Pointer<Type>& pointer);
};

// Task

class Task {
  public:
    Task();
    Task(unsigned int id);
    unsigned int id;
};
// Stream

#ifdef USE_CUDA
    #include "kmm.cuh"
#else
class Stream {
  public:
    Stream();
};
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
    void setSize(std::size_t size);
    // Manipulate device
    std::shared_ptr<DeviceType> getDevice();
    void setDevice(CPU& device);
    void setDevice(CUDA& device);
    // Manipulate special memory
    std::shared_ptr<MemoryType> getMemory();
    void setMemory(CUDAPinned& memory);
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

// Manager

class ManagerImpl;

class Manager {
  public:
    Manager();
    Task run();

    template<typename Type>
    Pointer<Type> create(std::size_t size) {
        return Pointer<Type>(this->create_impl(size));
    }

    template<typename Type>
    Pointer<Type> create(std::size_t size, CUDAPinned& memory) {
        return Pointer<Type>(this->create_impl(size, memory));
    }

    template<typename Type>
    void move_to(CPU& device, Pointer<Type>& pointer) {
        this->move_to_impl(device, pointer.id);
    }

    template<typename Type>
    void move_to(CUDA& device, Pointer<Type>& pointer) {
        this->move_to_impl(device, pointer.id);
    }

    template<typename Type>
    void release(Pointer<Type>& pointer) {
        release_impl(pointer.id);
    }

    template<typename Type>
    void copy_release(Pointer<Type>& pointer, void* target) {
        this->copy_release_impl(pointer.id, target);
    }

  private:
    unsigned int create_impl(size_t size);
    unsigned int create_impl(size_t size, CUDAPinned& memory);
    void move_to_impl(CPU& device, unsigned int pointer_id);
    void move_to_impl(CUDA& device, unsigned int pointer_id);
    void release_impl(unsigned int pointer_id);
    void copy_release_impl(unsigned int pointer_id, void* target);
    std::shared_ptr<ManagerImpl> impl_;
};

// Misc

inline bool is_cpu(CPU& device) {
    return true;
}

inline bool is_cpu(GPU& device) {
    return false;
}

inline bool is_cuda_pinned(Buffer& buffer) {
    if (dynamic_cast<CUDAPinned*>(buffer.getMemory().get()) != nullptr) {
        return true;
    }
    return false;
}

inline bool is_pinned(Buffer& buffer) {
    return is_cuda_pinned(buffer);
}

inline bool on_cpu(Buffer& buffer) {
    if (dynamic_cast<CPU*>(buffer.getDevice().get()) != nullptr) {
        return true;
    }
    return false;
}

inline bool on_cuda(Buffer& buffer) {
    if (dynamic_cast<CUDA*>(buffer.getDevice().get()) != nullptr) {
        return true;
    }
    return false;
}

inline bool same_device(CPU& device_one, CPU& device_two) {
    return true;
}

inline bool same_device(CPU& device_one, GPU& device_two) {
    return false;
}

inline bool same_device(GPU& device_one, CPU& device_two) {
    return false;
}

inline bool same_device(CUDA& device_one, CUDA& device_two) {
    return device_one.device_id == device_two.device_id;
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
Pointer<Type>::Pointer(unsigned int id) {
    this->id = id;
    this->type = Type();
    this->dirty = false;
}

// WritePointer

template<typename Type>
WritePointer<Type>::WritePointer(Pointer<Type>& pointer) {
    this->id = pointer.id;
    this->type = pointer.type;
    this->dirty = true;
    pointer.dirty = true;
}

}  // namespace kmm
