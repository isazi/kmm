#pragma once

#include <cstddef>
#include <map>

namespace kmm {

class DataType {};

class UInteger: public DataType {};

class Integer: public DataType {};

class FP_Single: public DataType {};

class FP_Double: public DataType {};

class DeviceType {};

class CPU: public DeviceType {};

class CUDA: public DeviceType {
  public:
    CUDA();
    CUDA(unsigned int device_id);
    unsigned int device_id;
};

class Pointer {
  public:
    Pointer();
    Pointer(unsigned int id);
    unsigned int id;
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
    // Allocate memory buffer
    void allocate(std::size_t size);
    // Allocate memory buffer using a Stream
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
    // Allocate buffer of size bytes on a GPU
    Pointer create(CUDA& device, std::size_t size);
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
