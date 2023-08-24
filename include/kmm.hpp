#pragma once

#include <cstddef>
#include <map>

namespace kmm {

class MemoryManager {
  public:
    MemoryManager();
    ~MemoryManager();
    // Allocate buffer of size bytes on the GPU
    unsigned int allocate(std::size_t size);
    // Allocate buffer of size bytes on the GPU, and copy the content of host_buffer to it
    unsigned int allocate(std::size_t size, void* host_buffer);
    // Copy the content of host_buffer to the GPU
    void copy_to(unsigned int device_buffer, std::size_t size, void* host_buffer);
    // Copy the content of GPU memory to host buffer
    void copy_from(unsigned int device_buffer, std::size_t size, void* host_buffer);
    // Free the memory on the GPU
    void release(unsigned int device_buffer);
    // Copy the content of GPU memory to the host and then free it
    void release(unsigned int device_buffer, std::size_t size, void* host_buffer);
    // Return a pointer to the used CUDA stream
    inline cudaStream_t* getStream();

  private:
    unsigned int next_allocation;
    cudaStream_t* stream;
    std::map<unsigned int, void*> allocations;
};

}  // namespace kmm
