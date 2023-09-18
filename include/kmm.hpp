#pragma once

#include <cstddef>
#include <map>
#include <memory>

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

class Pointer {
  public:
    Pointer();
    Pointer(unsigned int id);
    Pointer(unsigned int id, DataType& type);
    bool dirty;
    unsigned int id;
    DataType type;
};

class WritePointer: public Pointer {
  public:
    WritePointer(Pointer& pointer);
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
    ~Buffer();
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
    ~Manager();
    // Request memory
    Pointer create(std::size_t size, DataType& type);
    // Copy data from the host to a GPU
    void copy_to(CUDA& device, Pointer& device_pointer, Pointer& host_pointer);
    // Copy the content from a GPU to the host
    void copy_from(CUDA& device, Pointer& device_pointer, Pointer& host_pointer);
    // Release memory
    void release(Pointer& device_pointer);
    // Copy the content of GPU memory to the host and then free it
    void release(CUDA& device, Pointer& device_pointer, Pointer& host_pointer);
    // Execute a function on a device
    // TODO

  private:
    unsigned int next_allocation;
    std::map<unsigned int, Stream> streams;
    std::map<unsigned int, Buffer> allocations;
    // Check if a device has a stream allocated
    bool stream_exist(unsigned int stream);
};

}  // namespace kmm
