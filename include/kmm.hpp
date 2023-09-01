#pragma once

#include <cstddef>
#include <map>

namespace kmm {

enum DataType { UInteger, Integer, FP_Single, FP_Double };

enum DeviceType { Undefined, CPU, CUDA };

class Stream {
  public:
    Stream();
    Stream(DeviceType device);
    ~Stream();
    // Return a CUDA stream
    cudaStream_t cudaGetStream();

  private:
    DeviceType device;
    cudaStream_t cuda_stream;
};

class Buffer {
  public:
    Buffer();
    Buffer(DeviceType device);
    Buffer(DeviceType device, unsigned int device_id);
    ~Buffer();
    // Return true if the buffer is allocated
    bool is_allocated() const;
    // Allocate memory buffer
    void allocate(std::size_t size);
    // Allocate memory buffer using a Stream
    void allocate(std::size_t size, Stream& stream);
    // Destroy the allocate buffer
    void destroy();
    // Destroy the allocate buffer
    void destroy(Stream& stream);
    // Return a pointer to the allocated buffer
    void* getPointer();
    // Return the device type
    DeviceType getDeviceType();
    // Return the device id
    unsigned int getDeviceId();

  private:
    unsigned int device_id;
    void* buffer;
    DataType buffer_type;
    DeviceType device;
};

class Manager {
  public:
    Manager();
    ~Manager();
    // Allocate buffer of size bytes on a device
    unsigned int create(DeviceType device, std::size_t size);
    // Allocate buffer of size bytes on a device
    unsigned int create(DeviceType device, std::size_t size, unsigned int device_id);
    // Copy the content of host_buffer to the GPU
    void copy_to(
        DeviceType device,
        unsigned int device_buffer,
        std::size_t size,
        void* host_buffer,
        unsigned int device_id);
    // Copy the content of GPU memory to host buffer
    void copy_from(
        DeviceType device,
        unsigned int device_buffer,
        std::size_t size,
        void* host_buffer,
        unsigned int device_id);
    // Free the allocation
    void release(unsigned int device_buffer);
    // Free the allocation
    void release(unsigned int device_buffer, unsigned int device_id);
    // Copy the content of GPU memory to the host and then free it
    void release(
        DeviceType device,
        unsigned int device_buffer,
        std::size_t size,
        void* host_buffer,
        unsigned int device_id);

  private:
    unsigned int next_allocation;
    std::map<unsigned int, Stream> streams;
    std::map<unsigned int, Buffer> allocations;
};

}  // namespace kmm
