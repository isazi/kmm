#include <stdexcept>
#include <string>

#include "kmm.hpp"

namespace kmm {

inline void cudaErrorCheck(cudaError_t err, std::string message) {
    if (err != cudaSuccess) {
        throw std::runtime_error(message);
    }
}

Manager::Manager() {
    this->next_allocation = 0;
    this->allocations = std::map<unsigned int, Buffer>();
    this->streams = std::map<unsigned int, Stream>();
}

Manager::~Manager() {}

unsigned int Manager::create(DeviceType device, std::size_t size) {
    unsigned int allocation_id;

    allocation_id = this->next_allocation++;
    this->allocations[allocation_id] = Buffer(device);
    switch (device) {
        case CPU:
            this->allocations[allocation_id].allocate(size);
            break;

        default:
            break;
    }

    return allocation_id;
}

unsigned int Manager::create(DeviceType device, std::size_t size, unsigned int device_id) {
    unsigned int allocation_id;

    allocation_id = this->next_allocation++;
    this->allocations[allocation_id] = Buffer(device);
    switch (device) {
        case CUDA:
            if (this->streams.find(device_id) == this->streams.end()) {
                this->streams[device_id] = Stream(device);
            }
            this->allocations[allocation_id].allocate(size, this->streams[device_id]);

        default:
            break;
    }

    return allocation_id;
}

void Manager::copy_to(
    DeviceType device,
    unsigned int device_buffer,
    std::size_t size,
    void* host_buffer,
    unsigned int device_id) {
    cudaError_t err = cudaSuccess;

    switch (device) {
        case CUDA:
            err = cudaMemcpyAsync(
                this->allocations[device_buffer].getPointer(),
                host_buffer,
                size,
                cudaMemcpyHostToDevice,
                this->streams[device_id]);
            cudaErrorCheck(err, "Impossible to copy memory to device.");
            break;

        default:
            break;
    }
}

void Manager::copy_from(
    DeviceType device,
    unsigned int device_buffer,
    std::size_t size,
    void* host_buffer,
    unsigned int device_id) {
    cudaError_t err = cudaSuccess;

    switch (device) {
        case CUDA:
            err = cudaMemcpyAsync(
                host_buffer,
                this->allocations[device_buffer].getPointer(),
                size,
                cudaMemcpyDeviceToHost,
                this->streams[device_id]);
            cudaErrorCheck(err, "Impossible to copy memory to host.");
            break;

        default:
            break;
    }
}

void Manager::release(unsigned int device_buffer) {
    cudaError_t err = cudaSuccess;

    err = cudaFreeAsync(this->allocations[device_buffer], this->stream);
    cudaErrorCheck(err, "Impossible to release memory.");
    this->allocations[device_buffer] = nullptr;
}

void Manager::release(unsigned int device_buffer, std::size_t size, void* host_buffer) {
    this->copy_from(device_buffer, size, host_buffer);
    this->release(device_buffer);
}

Buffer::Buffer() {
    this->device = Undefined;
    this->device_id = 0;
    this->buffer = nullptr;
}

Buffer::Buffer(DeviceType device) {
    this->device = device;
    this->device_id = 0;
    this->buffer = nullptr;
}

Buffer::Buffer(DeviceType device, unsigned int device_id) {
    this->device = device;
    this->device_id = device_id;
    this->buffer = nullptr;
}

Buffer::~Buffer() {}

bool Buffer::is_allocated() const {
    return this->buffer != nullptr;
}

void Buffer::allocate(std::size_t size) {
    switch (this->device) {
        case CPU:
            this->buffer = malloc(size);
            break;

        case CUDA:
            cudaError_t err = cudaMalloc(&(this->buffer), size);
            cudaErrorCheck(err, "Impossible to allocate CUDA memory.");

        default:
            break;
    }
}

void Buffer::allocate(std::size_t size, Stream& stream) {
    switch (this->device) {
        case CUDA:
            cudaError_t err = cudaMallocAsync(&(this->buffer), size, stream.cudaGetStream());
            cudaErrorCheck(err, "Impossible to allocate CUDA memory.");

        default:
            break;
    }
}

void Buffer::destroy() {
    switch (this->device) {
        case CPU:
            free(this->buffer);
            break;

        case CUDA:
            auto err = cudaFree(this->buffer);
            cudaErrorCheck(err, "Impossible to release memory.");
            break;

        default:
            break;
    }
}

void Buffer::destroy(Stream& stream) {
    switch (this->device) {
        case CUDA:
            auto err = cudaFreeAsync(this->buffer, this->stream);
            cudaErrorCheck(err, "Impossible to release memory.");
            break;

        default:
            break;
    }
}

void* Buffer::getPointer() {
    return this->buffer;
}

DeviceType Buffer::getDeviceType() {
    return this->device;
}

unsigned int Buffer::getDeviceId() {
    return this->device_id;
}

Stream::Stream() {
    this->device = Undefined;
    this->cuda_stream = nullptr;
}

Stream::Stream(DeviceType device) {
    this->device = device;
    this->cuda_stream = nullptr;

    switch (this->device) {
        case CUDA:
            auto err = cudaStreamCreate(&(this->cuda_stream));
            cudaErrorCheck(err, "Impossible to create CUDA stream.");
            break;

        default:
            break;
    }
}

Stream::~Stream() {
    switch (this->device) {
        case CUDA:
            auto err = cudaStreamDestroy(this->cuda_stream);
            cudaErrorCheck(err, "Impossible to destroy CUDA stream.");
            break;

        default:
            break;
    }
}

cudaStream_t Stream::cudaGetStream() {
    return this->cuda_stream;
}

}  // namespace kmm
