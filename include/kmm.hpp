#pragma once

#include <cstddef>
#include <map>

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

class CPU: public DeviceType {};

class GPU: public DeviceType {
  public:
    GPU();
    GPU(unsigned int device_id);
    unsigned int device_id;
};

class CUDA: public GPU {};

class Pointer {
  public:
    Pointer();
    Pointer(unsigned int id);
    Pointer(unsigned int id, DataType& type);
    unsigned int id;
    DataType type;
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
    Buffer(DeviceType& device);
    ~Buffer();
    // Return true if the buffer is allocated
    bool is_allocated() const;
    // Allocate memory buffer on the host
    void allocate(std::size_t size);
    // Allocate memory buffer on a GPU
    void allocate(CUDA& device, std::size_t size, Stream& stream);
    // Destroy the allocate buffer
    void destroy();
    // Destroy the allocate buffer
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
};

class Manager {
  public:
    Manager();
    ~Manager();
    // Allocate buffer of size bytes on the host
    Pointer create(CPU& device, std::size_t size);
    Pointer create(CPU& device, std::size_t size, DataType& type);
    // Allocate buffer of size bytes on a GPU
    Pointer create(CUDA& device, std::size_t size);
    Pointer create(CUDA& device, std::size_t size, DataType& type);
    // Copy data from the host to a GPU
    void copy_to(CUDA& device, Pointer& device_buffer, std::size_t size, Pointer& host_buffer);
    // Copy the content of GPU memory to host buffer
    void copy_from(CUDA& device, Pointer& device_buffer, std::size_t size, Pointer& host_buffer);
    // Free the allocation
    void release(Pointer& device_buffer);
    // Free the allocation
    void release(CUDA& device, Pointer& device_buffer);
    // Copy the content of GPU memory to the host and then free it
    void release(CUDA& device, Pointer& device_buffer, std::size_t size, Pointer& host_buffer);
    // Execute a function on a device
    // TODO

  private:
    unsigned int next_allocation;
    std::map<unsigned int, Stream> streams;
    std::map<unsigned int, Buffer> allocations;
    bool stream_exist(unsigned int stream);
};

}  // namespace kmm
