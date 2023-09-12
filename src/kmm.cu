#include <iostream>
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

// Manager

Manager::Manager() {
    this->next_allocation = 0;
    this->allocations = std::map<unsigned int, Buffer>();
    this->streams = std::map<unsigned int, Stream>();
}

Manager::~Manager() {}

Pointer Manager::create(CPU& device, std::size_t size) {
    unsigned int allocation_id;

    allocation_id = this->next_allocation++;
    this->allocations[allocation_id] = Buffer(device);
    this->allocations[allocation_id].allocate(size);

    return Pointer(allocation_id);
}

Pointer Manager::create(CUDA& device, std::size_t size) {
    unsigned int allocation_id;

    allocation_id = this->next_allocation++;
    this->allocations[allocation_id] = Buffer(device);
    if (!this->stream_exist(device.device_id)) {
        this->streams[device.device_id] = Stream(device);
    }
    this->allocations[allocation_id].allocate(device, size, this->streams[device.device_id]);

    return Pointer(allocation_id);
}

void Manager::copy_to(
    CUDA& device,
    Pointer& device_buffer,
    std::size_t size,
    Pointer& host_buffer) {
    auto err = cudaMemcpyAsync(
        this->allocations[device_buffer.id].getPointer(),
        this->allocations[host_buffer.id].getPointer(),
        size,
        cudaMemcpyHostToDevice,
        this->streams[device.device_id].getStream(device));
    cudaErrorCheck(err, "Impossible to copy memory to device.");
}

void Manager::copy_from(
    CUDA& device,
    Pointer& device_buffer,
    std::size_t size,
    Pointer& host_buffer) {
    auto err = cudaMemcpyAsync(
        this->allocations[host_buffer.id].getPointer(),
        this->allocations[device_buffer.id].getPointer(),
        size,
        cudaMemcpyDeviceToHost,
        this->streams[device.device_id].getStream(device));
    cudaErrorCheck(err, "Impossible to copy memory to host.");
}

void Manager::release(Pointer& device_buffer) {
    this->allocations[device_buffer.id].destroy();
}

void Manager::release(CUDA& device, Pointer& device_buffer) {
    this->allocations[device_buffer.id].destroy(device, this->streams[device.device_id]);
}

void Manager::release(
    CUDA& device,
    Pointer& device_buffer,
    std::size_t size,
    Pointer& host_buffer) {
    this->copy_from(device, device_buffer, size, host_buffer);
    this->release(device, device_buffer);
}

// Buffer

Buffer::Buffer() {
    this->buffer = nullptr;
}

Buffer::Buffer(DeviceType& device) {
    this->buffer = nullptr;
}

Buffer::~Buffer() {}

bool Buffer::is_allocated() const {
    return this->buffer != nullptr;
}

void Buffer::allocate(std::size_t size) {
    this->buffer = malloc(size);
}

void Buffer::allocate(CUDA& device, std::size_t size, Stream& stream) {
    auto err = cudaMallocAsync(&(this->buffer), size, stream.getStream(device));
    cudaErrorCheck(err, "Impossible to allocate CUDA memory.");
}

void Buffer::destroy() {
    free(this->buffer);
    this->buffer = nullptr;
}

void Buffer::destroy(CUDA& device, Stream& stream) {
    auto err = cudaFreeAsync(this->buffer, stream.getStream(device));
    cudaErrorCheck(err, "Impossible to release memory.");
    this->buffer = nullptr;
}

void* Buffer::getPointer() {
    return this->buffer;
}

unsigned int* Buffer::getPointer(UInteger& type) {
    return reinterpret_cast<unsigned int*>(this->buffer);
}

int* Buffer::getPointer(Integer& type) {
    return reinterpret_cast<int*>(this->buffer);
}

float* Buffer::getPointer(FP_Single& type) {
    return reinterpret_cast<float*>(this->buffer);
}

double* Buffer::getPointer(FP_Double& type) {
    return reinterpret_cast<double*>(this->buffer);
}

//Stream

Stream::Stream() {
    this->cuda_stream = nullptr;
}

Stream::Stream(CUDA& device) {
    try {
        auto err = cudaStreamCreate(&(this->cuda_stream));
        cudaErrorCheck(err, "Impossible to create CUDA stream.");
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        this->cuda_stream = nullptr;
    }
}

Stream::~Stream() {
    if (this->cuda_stream != nullptr) {
        try {
            auto err = cudaStreamDestroy(this->cuda_stream);
            cudaErrorCheck(err, "Impossible to destroy CUDA stream.");
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
    }
}

cudaStream_t Stream::getStream(CUDA& device) {
    return this->cuda_stream;
}

// Pointer

Pointer::Pointer() {}

Pointer::Pointer(unsigned int id) {
    this->id = id;
}

// CUDA

CUDA::CUDA() {
    this->device_id = 0;
}

CUDA::CUDA(unsigned int device_id) {
    this->device_id = device_id;
}

}  // namespace kmm
