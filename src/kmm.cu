#include <stdexcept>
#include <string>

#include "kmm.hpp"

namespace kmm {

inline void cudaErrorCheck(cudaError_t err, std::string message) {
    if (err != cudaSuccess) {
        throw std::runtime_error(message);
    }
}

bool Manager::stream_exist(unsigned int stream) {
    return this->streams.find(stream) != this->streams.end();
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

unsigned int Manager::create(DeviceType device, unsigned int device_id, std::size_t size) {
    unsigned int allocation_id;

    allocation_id = this->next_allocation++;
    this->allocations[allocation_id] = Buffer(device);
    switch (device) {
        case CUDA:
            if (!this->stream_exist(device_id)) {
                this->streams[device_id] = Stream(device);
            }
            this->allocations[allocation_id].allocate(size, this->streams[device_id]);
            break;

        default:
            break;
    }

    return allocation_id;
}

void Manager::copy_to(
    DeviceType device,
    unsigned int device_buffer,
    std::size_t size,
    unsigned int host_buffer,
    unsigned int device_id) {
    cudaError_t err = cudaSuccess;

    switch (device) {
        case CUDA:
            err = cudaMemcpyAsync(
                this->allocations[device_buffer].getPointer(),
                this->allocations[host_buffer].getPointer(),
                size,
                cudaMemcpyHostToDevice,
                this->streams[device_id].cudaGetStream());
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
    unsigned int host_buffer,
    unsigned int device_id) {
    cudaError_t err = cudaSuccess;

    switch (device) {
        case CUDA:
            err = cudaMemcpyAsync(
                this->allocations[host_buffer].getPointer(),
                this->allocations[device_buffer].getPointer(),
                size,
                cudaMemcpyDeviceToHost,
                this->streams[device_id].cudaGetStream());
            cudaErrorCheck(err, "Impossible to copy memory to host.");
            break;

        default:
            break;
    }
}

void Manager::release(unsigned int device_buffer) {
    this->allocations[device_buffer].destroy();
}

void Manager::release(unsigned int device_buffer, unsigned int device_id) {
    this->allocations[device_buffer].destroy(this->streams[device_id]);
}

void Manager::release(
    DeviceType device,
    unsigned int device_buffer,
    std::size_t size,
    unsigned int host_buffer,
    unsigned int device_id) {
    this->copy_from(device, device_buffer, size, host_buffer, device_id);
    this->release(device_buffer, device_id);
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
    cudaError_t err = cudaSuccess;

    switch (this->device) {
        case CPU:
            this->buffer = malloc(size);
            break;

        default:
            break;
    }
}

void Buffer::allocate(std::size_t size, Stream& stream) {
    cudaError_t err = cudaSuccess;

    switch (this->device) {
        case CUDA:
            err = cudaMallocAsync(&(this->buffer), size, stream.cudaGetStream());
            cudaErrorCheck(err, "Impossible to allocate CUDA memory.");
            break;

        default:
            break;
    }
}

void Buffer::destroy() {
    cudaError_t err = cudaSuccess;

    switch (this->device) {
        case CPU:
            free(this->buffer);
            break;

        default:
            break;
    }
    this->buffer = nullptr;
}

void Buffer::destroy(Stream& stream) {
    cudaError_t err = cudaSuccess;

    switch (this->device) {
        case CUDA:
            err = cudaFreeAsync(this->buffer, stream.cudaGetStream());
            cudaErrorCheck(err, "Impossible to release memory.");
            break;

        default:
            break;
    }
    this->buffer = nullptr;
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
    cudaError_t err = cudaSuccess;

    switch (this->device) {
        case CUDA:
            err = cudaStreamCreate(&(this->cuda_stream));
            cudaErrorCheck(err, "Impossible to create CUDA stream.");
            break;

        default:
            break;
    }
}

Stream::~Stream() {
    cudaError_t err = cudaSuccess;

    switch (this->device) {
        case CUDA:
            err = cudaStreamDestroy(this->cuda_stream);
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
