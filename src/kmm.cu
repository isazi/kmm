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

Pointer Manager::create(std::size_t size, UInteger& type) {
    unsigned int allocation_id;

    allocation_id = this->next_allocation++;
    this->allocations[allocation_id] = Buffer(size);

    return Pointer(allocation_id, type);
}

Pointer Manager::create(std::size_t size, Integer& type) {
    unsigned int allocation_id;

    allocation_id = this->next_allocation++;
    this->allocations[allocation_id] = Buffer(size);

    return Pointer(allocation_id, type);
}

Pointer Manager::create(std::size_t size, FP_Single& type) {
    unsigned int allocation_id;

    allocation_id = this->next_allocation++;
    this->allocations[allocation_id] = Buffer(size);

    return Pointer(allocation_id, type);
}

Pointer Manager::create(std::size_t size, FP_Double& type) {
    unsigned int allocation_id;

    allocation_id = this->next_allocation++;
    this->allocations[allocation_id] = Buffer(size);

    return Pointer(allocation_id, type);
}

void Manager::copy_to(CUDA& device, Pointer& device_pointer, Pointer& host_pointer) {
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

void Manager::copy_from(CUDA& device, Pointer& device_pointer, Pointer& host_pointer) {
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

void Manager::release(Pointer& device_pointer) {
    this->allocations[device_pointer.id].destroy();
}

void Manager::release(CUDA& device, Pointer& device_pointer, Pointer& host_buffer) {
    this->copy_from(device, device_pointer, host_buffer);
    this->release(device_pointer);
}

// Buffer

Buffer::Buffer() {
    this->buffer = nullptr;
    this->size = 0;
    this->device = std::shared_ptr<UnknownDevice>();
}

Buffer::Buffer(std::size_t size) {
    this->buffer = nullptr;
    this->size = size;
    this->device = std::shared_ptr<UnknownDevice>();
}

Buffer::Buffer(CPU& device, std::size_t size) {
    this->buffer = nullptr;
    this->size = size;
    this->device = std::shared_ptr<CPU>(&device);
}

Buffer::Buffer(CUDA& device, std::size_t size) {
    this->buffer = nullptr;
    this->size = size;
    this->device = std::shared_ptr<CUDA>(&device);
}

Buffer::~Buffer() {}

std::size_t Buffer::getSize() const {
    return this->size;
}

void Buffer::setSize(std::size_t size) {
    this->size = size;
}

std::shared_ptr<DeviceType> Buffer::getDevice() {
    return this->device;
}

void Buffer::setDevice(CPU& device) {
    this->device = std::shared_ptr<CPU>(&device);
}

void Buffer::setDevice(CUDA& device) {
    this->device = std::shared_ptr<CUDA>(&device);
}

bool Buffer::is_allocated() const {
    return this->buffer != nullptr;
}

bool Buffer::is_allocated(CUDA& device) const {
    return (this->buffer != nullptr)
        && (device.device_id == dynamic_cast<CUDA*>(this->device.get())->device_id);
}

void Buffer::allocate() {
    this->buffer = malloc(this->size);
}

void Buffer::allocate(CUDA& device, Stream& stream) {
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

Pointer::Pointer(unsigned int id, UInteger& type) {
    this->id = id;
    this->type = std::shared_ptr<UInteger>(&type);
    this->dirty = false;
}

Pointer::Pointer(unsigned int id, Integer& type) {
    this->id = id;
    this->type = std::shared_ptr<Integer>(&type);
    this->dirty = false;
}

Pointer::Pointer(unsigned int id, FP_Single& type) {
    this->id = id;
    this->type = std::shared_ptr<FP_Single>(&type);
    this->dirty = false;
}

Pointer::Pointer(unsigned int id, FP_Double& type) {
    this->id = id;
    this->type = std::shared_ptr<FP_Double>(&type);
    this->dirty = false;
}

// WritePointer

WritePointer::WritePointer(Pointer& pointer) {
    this->id = pointer.id;
    this->type = pointer.type;
    pointer.dirty = true;
}

// GPU

GPU::GPU() {
    this->device_id = 0;
}

GPU::GPU(unsigned int device_id) {
    this->device_id = device_id;
}

}  // namespace kmm
