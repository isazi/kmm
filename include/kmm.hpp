#include <cstddef>
#include <map>
#include <memory>

#pragma once

namespace kmm {

class DataType {
  public:
    virtual ~DataType() = default;
};

class UInteger: public DataType {};

class Integer: public DataType {};

class FP_Single: public DataType {};

class FP_Double: public DataType {};

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

class CUDA: public GPU {};

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

class Stream {
  public:
    Stream();
    Stream(CUDA& device);
    ~Stream();
    // Return a CUDA stream
    cudaStream_t getStream(CUDA& device);

  private:
    cudaStream_t cuda_stream;
};

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
    bool is_allocated(CUDA& device) const;
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

class Manager {
  public:
    Manager();
    // Request memory
    template<typename Type>
    Pointer<Type> create(std::size_t size);
    // Copy data from the host to a GPU
    template<typename Type>
    void copy_to(CUDA& device, Pointer<Type>& device_pointer, Pointer<Type>& host_pointer);
    // Copy the content from a GPU to the host
    template<typename Type>
    void copy_from(CUDA& device, Pointer<Type>& device_pointer, Pointer<Type>& host_pointer);
    // Release memory
    template<typename Type>
    void release(Pointer<Type>& device_pointer);
    template<typename Type>
    void release(CUDA& device, Pointer<Type>& device_pointer, Pointer<Type>& host_pointer);
    // Execute a function on a device
    // TODO

  private:
    unsigned int next_allocation;
    std::map<unsigned int, Stream> streams;
    std::map<unsigned int, Buffer> allocations;
    // Check if a device has a stream allocated
    bool stream_exist(unsigned int stream);
};

inline void cudaErrorCheck(cudaError_t err, std::string message) {
    if (err != cudaSuccess) {
        throw std::runtime_error(message);
    }
}

// Manager

template<typename Type>
Pointer<Type> Manager::create(std::size_t size) {
    unsigned int allocation_id;

    allocation_id = this->next_allocation++;
    this->allocations[allocation_id] = Buffer(size);

    return Pointer<Type>(allocation_id);
}

template<typename Type>
void Manager::copy_to(CUDA& device, Pointer<Type>& device_pointer, Pointer<Type>& host_pointer) {
    auto device_buffer = this->allocations[device_pointer.id];
    auto host_buffer = this->allocations[host_pointer.id];
    auto stream = this->streams[device.device_id];

    if (!device_buffer.is_allocated()) {
        device_buffer.allocate(device, stream);
    } else if (!device_buffer.is_allocated(device)) {
        device_buffer.destroy();
        device_buffer.allocate(device, stream);
    }

    auto err = cudaMemcpyAsync(
        device_buffer.getPointer(),
        host_buffer.getPointer(),
        device_buffer.getSize(),
        cudaMemcpyHostToDevice,
        stream.getStream(device));
    cudaErrorCheck(err, "Impossible to copy memory to device.");
}

template<typename Type>
void Manager::copy_from(CUDA& device, Pointer<Type>& device_pointer, Pointer<Type>& host_pointer) {
    auto device_buffer = this->allocations[device_pointer.id];
    auto host_buffer = this->allocations[host_pointer.id];
    auto stream = this->streams[device.device_id];

    if (!host_buffer.is_allocated()) {
        host_buffer.allocate();
    }

    auto err = cudaMemcpyAsync(
        host_buffer.getPointer(),
        device_buffer.getPointer(),
        host_buffer.getSize(),
        cudaMemcpyDeviceToHost,
        stream.getStream(device));
    cudaErrorCheck(err, "Impossible to copy memory to host.");
}

template<typename Type>
void Manager::release(Pointer<Type>& device_pointer) {
    this->allocations[device_pointer.id].destroy();
}

template<typename Type>
void Manager::release(CUDA& device, Pointer<Type>& device_pointer, Pointer<Type>& host_buffer) {
    this->copy_from(device, device_pointer, host_buffer);
    this->release(device_pointer);
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
