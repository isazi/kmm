#include <stdexcept>
#include <string>

#include "kmm.hpp"

namespace kmm {
inline void cudaErrorCheck(cudaError_t err, std::string message) {
    if (err != cudaSuccess) {
        throw std::runtime_error(message);
    }
}

MemoryManager::MemoryManager() {
    this->next_allocation = 0;
    this->allocations = std::map<unsigned int, void*>();
    this->stream = nullptr;
}

MemoryManager::~MemoryManager() {
    for (auto const& [allocation_id, device_buffer] : this->allocations) {
        if (device_buffer != nullptr) {
            this->release(allocation_id);
        }
    }
    if (this->stream != nullptr) {
        cudaStreamDestroy(this->stream);
    }
}

unsigned int MemoryManager::allocate(std::size_t size) {
    cudaError_t err = cudaSuccess;
    unsigned int allocation_id;

    if (!(this->stream)) {
        err = cudaStreamCreate(&(this->stream));
        cudaErrorCheck(err, "Impossible to create stream.");
    }

    allocation_id = this->next_allocation++;
    this->allocations[allocation_id] = nullptr;
    err = cudaMallocAsync(&(this->allocations[allocation_id]), size, this->stream);
    cudaErrorCheck(err, "Impossible to allocate memory.");

    return allocation_id;
}

unsigned int MemoryManager::allocate(std::size_t size, void* host_buffer) {
    unsigned int allocation_id;

    allocation_id = this->allocate(size);
    this->copy_to(allocation_id, size, host_buffer);

    return allocation_id;
}

void MemoryManager::copy_to(unsigned int device_buffer, std::size_t size, void* host_buffer) {
    cudaError_t err = cudaSuccess;

    err = cudaMemcpyAsync(
        this->allocations[device_buffer],
        host_buffer,
        size,
        cudaMemcpyHostToDevice,
        this->stream);
    cudaErrorCheck(err, "Impossible to copy memory to device.");
}

void MemoryManager::copy_from(unsigned int device_buffer, std::size_t size, void* host_buffer) {
    cudaError_t err = cudaSuccess;

    err = cudaMemcpyAsync(
        host_buffer,
        this->allocations[device_buffer],
        size,
        cudaMemcpyDeviceToHost,
        this->stream);
    cudaErrorCheck(err, "Impossible to copy memory to host.");
}

void MemoryManager::release(unsigned int device_buffer) {
    cudaError_t err = cudaSuccess;

    err = cudaFreeAsync(this->allocations[device_buffer], this->stream);
    cudaErrorCheck(err, "Impossible to release memory.");
    this->allocations[device_buffer] = nullptr;
}

void MemoryManager::release(unsigned int device_buffer, std::size_t size, void* host_buffer) {
    this->copy_from(device_buffer, size, host_buffer);
    this->release(device_buffer);
}

}  // namespace kmm
