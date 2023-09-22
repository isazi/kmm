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
    Buffer(CPU& device, std::size_t size);
    Buffer(CUDA& device, std::size_t size);
    // Manipulate size
    std::size_t getSize() const;
    void setSize(std::size_t size);
    // Manipulate device
    std::shared_ptr<DeviceType> getDevice();
    void setDevice(CPU& device);
    void setDevice(CUDA& device);
    // Return true if the buffer is allocated
    bool is_allocated() const;
    // Allocate memory
    void allocate();
    void allocate(CUDA& device, Stream& stream);
    // Free memory
    void destroy();
    void destroy(CUDA& device, Stream& stream);
    // Return a pointer to the allocated buffer
    void* getPointer();
    // Return a typed pointer
    unsigned int* getPointer(UInteger& type);
    int* getPointer(Integer& type);
    float* getPointer(FP_Single& type);
    double* getPointer(FP_Double& type);

  private:
    void* buffer;
    std::size_t size;
    std::shared_ptr<DeviceType> device;
};

// Manager

class Manager {
  public:
    Manager();
    // Request memory
    template<typename Type>
    Pointer<Type> create(std::size_t size);
    // Move data
    template<typename Type>
    void move_to(CPU& device, Pointer<Type>& pointer);
    template<typename Type>
    void move_to(CUDA& device, Pointer<Type>& pointer);
    // Release memory
    template<typename Type>
    void release(Pointer<Type>& pointer);
    template<typename Type>
    void copy_release(Pointer<Type>& pointer, void* target);
    // Execute a function on a device
    // TODO

  private:
    unsigned int next_allocation;
    std::map<unsigned int, Stream> streams;
    std::map<unsigned int, Buffer> allocations;
    // Check if a device has a stream allocated
    bool stream_exist(unsigned int stream);
};

// Misc

inline bool is_cpu(CPU& device) {
    return true;
}

inline bool is_cpu(GPU& device) {
    return false;
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

template<typename Type>
Pointer<Type> Manager::create(std::size_t size) {
    unsigned int allocation_id;

    allocation_id = this->next_allocation++;
    this->allocations[allocation_id] = Buffer(size);

    return Pointer<Type>(allocation_id);
}

template<typename Type>
void Manager::move_to(CPU& device, Pointer<Type>& pointer) {
    auto source_buffer = this->allocations[pointer.id];

    if (on_cpu(source_buffer)) {
        return;
    }
    if (source_buffer.is_allocated()) {
        auto source_device = *(dynamic_cast<CUDA*>(source_buffer.getDevice().get()));
        auto target_buffer = Buffer(device, source_buffer.getSize());
        auto stream = this->streams[source_device.device_id];
        target_buffer.allocate();
        cudaCopyD2H(source_device, source_buffer, target_buffer, stream);
        source_buffer.destroy(source_device, stream);
        this->allocations[pointer.id] = target_buffer;
    } else {
        source_buffer.setDevice(device);
    }
}

template<typename Type>
void Manager::move_to(CUDA& device, Pointer<Type>& pointer) {
    auto source_buffer = this->allocations[pointer.id];

    if (!source_buffer.is_allocated()) {
        source_buffer.setDevice(device);
        return;
    }
    auto target_buffer = Buffer(device, source_buffer.getSize());
    if (on_cuda(source_buffer)) {
        // source_buffer is allocated on a CUDA GPU
        auto source_device = *(dynamic_cast<CUDA*>(source_buffer.getDevice().get()));
        if (same_device(device, source_device)) {
            return;
        }
        auto stream = this->streams[source_device.device_id];
        target_buffer.allocate(device, stream);
        cudaCopyD2D(device, source_buffer, target_buffer, stream);
        source_buffer.destroy(source_device, stream);
    } else if (on_cpu(source_buffer)) {
        // source_buffer is allocated on a CPU
        auto stream = this->streams[device.device_id];
        target_buffer.allocate(device, stream);
        cudaCopyH2D(device, source_buffer, target_buffer, stream);
        source_buffer.destroy();
    }
    this->allocations[pointer.id] = target_buffer;
}

template<typename Type>
void Manager::release(Pointer<Type>& pointer) {
    auto buffer = this->allocations[pointer.id];

    if (on_cpu(buffer)) {
        buffer.destroy();
    } else if (on_cuda(buffer)) {
        auto device = *(dynamic_cast<CUDA*>(buffer.getDevice().get()));
        auto stream = this->streams[device.device_id];
        buffer.destroy(device, stream);
    }
}

template<typename Type>
void Manager::copy_release(Pointer<Type>& pointer, void* target) {
    auto buffer = this->allocations[pointer.id];

    if (on_cpu(buffer)) {
        memcpy(target, buffer.getPointer(), buffer.getSize());
    } else if (on_cuda(buffer)) {
        auto device = *(dynamic_cast<CUDA*>(buffer.getDevice().get()));
        auto stream = this->streams[device.device_id];
        cudaCopyD2H(device, buffer, target, stream);
    }
    this->release(pointer);
}

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
