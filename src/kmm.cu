#include <iostream>
#include <stdexcept>
#include <string>

#include "kmm.hpp"

namespace kmm {

bool Manager::stream_exist(unsigned int stream) {
    return this->streams.find(stream) != this->streams.end();
}

// Manager

Manager::Manager() {
    this->next_allocation = 0;
    this->allocations = std::map<unsigned int, Buffer>();
    this->streams = std::map<unsigned int, Stream>();
}

// Buffer

Buffer::Buffer() {
    this->buffer = nullptr;
    this->size = 0;
    this->device = std::make_shared<UnknownDevice>();
}

Buffer::Buffer(std::size_t size) {
    this->buffer = nullptr;
    this->size = size;
    this->device = std::make_shared<UnknownDevice>();
}

Buffer::Buffer(CPU& device, std::size_t size) {
    this->buffer = nullptr;
    this->size = size;
    this->device = std::make_shared<CPU>();
}

Buffer::Buffer(CUDA& device, std::size_t size) {
    this->buffer = nullptr;
    this->size = size;
    this->device = std::make_shared<CUDA>(device.device_id);
}

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
    this->device = std::make_shared<CPU>();
}

void Buffer::setDevice(CUDA& device) {
    this->device = std::make_shared<CUDA>(device.device_id);
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

// GPU

GPU::GPU() {
    this->device_id = 0;
}

GPU::GPU(unsigned int device_id) {
    this->device_id = device_id;
}

}  // namespace kmm
