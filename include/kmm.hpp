#include <cstddef>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#pragma once

namespace kmm {

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

    const std::type_info& type() const {
        return typeid(Type);
    }

    unsigned int id() const {
        return id_;
    }

    bool is_dirty() const {
        return dirty_;
    }

  protected:
    bool dirty_;
    unsigned int id_;
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
    void move_to(CPU& device, const Pointer<Type>& pointer) {
        this->move_to_impl(device, pointer.id());
    }

    template<typename Type>
    void move_to(CUDA& device, const Pointer<Type>& pointer) {
        this->move_to_impl(device, pointer.id());
    }

    template<typename Type>
    void release(const Pointer<Type>& pointer) {
        release_impl(pointer.id());
    }

    template<typename Type>
    void copy_release(const Pointer<Type>& pointer, void* target) {
        this->copy_release_impl(pointer.id(), target);
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

inline bool is_cpu(DeviceType& device) {
    return dynamic_cast<const CPU*>(&device) != nullptr;
}

inline bool is_cuda_pinned(Buffer& buffer) {
    return dynamic_cast<CUDAPinned*>(buffer.getMemory().get()) != nullptr;
}

inline bool is_pinned(Buffer& buffer) {
    return is_cuda_pinned(buffer);
}

inline bool on_cpu(Buffer& buffer) {
    return dynamic_cast<CPU*>(buffer.getDevice().get()) != nullptr;
}

inline bool on_cuda(Buffer& buffer) {
    return dynamic_cast<CUDA*>(buffer.getDevice().get()) != nullptr;
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
Pointer<Type>::Pointer(unsigned int id) {
    this->id_ = id;
    this->dirty_ = false;
}

// WritePointer

template<typename Type>
WritePointer<Type>::WritePointer(Pointer<Type>& pointer) {
    this->id_ = pointer.id();
    this->dirty_ = true;
    //    pointer.dirty_ = true;
}

}  // namespace kmm
