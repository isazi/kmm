#include "kmm/kmm.hpp"

namespace kmm {

// Misc

inline void cudaErrorCheck(cudaError_t err, std::string message) {
    if (err != cudaSuccess) {
        throw std::runtime_error(message);
    }
}

void cudaCopyD2H(CUDA& device, Buffer& source, Buffer& target, Stream& stream) {
    auto err = cudaMemcpyAsync(
        target.getPointer(),
        source.getPointer(),
        target.size(),
        cudaMemcpyDeviceToHost,
        stream.getStream(device));
    cudaErrorCheck(err, "Impossible to copy memory from device to host.");
}

void cudaCopyD2H(CUDA& device, Buffer& source, void* target, Stream& stream) {
    auto err = cudaMemcpyAsync(
        target,
        source.getPointer(),
        source.size(),
        cudaMemcpyDeviceToHost,
        stream.getStream(device));
    cudaErrorCheck(err, "Impossible to copy memory from device to host.");
}

void cudaCopyH2D(CUDA& device, Buffer& source, Buffer& target, Stream& stream) {
    auto err = cudaMemcpyAsync(
        target.getPointer(),
        source.getPointer(),
        target.size(),
        cudaMemcpyHostToDevice,
        stream.getStream(device));
    cudaErrorCheck(err, "Impossible to copy memory from host to device.");
}

void cudaCopyD2D(CUDA& device, Buffer& source, Buffer& target, Stream& stream) {
    auto err = cudaMemcpyAsync(
        target.getPointer(),
        source.getPointer(),
        target.size(),
        cudaMemcpyDeviceToDevice,
        stream.getStream(device));
    cudaErrorCheck(err, "Impossible to copy memory from device to device.");
}

// Buffer

void Buffer::allocate(CUDA& device, Stream& stream) {
    auto err = cudaMallocAsync(&(this->buffer_), size_, stream.getStream(device));
    cudaErrorCheck(err, "Impossible to allocate CUDA memory.");
}

void Buffer::allocate(CUDAPinned& memory) {
    auto err = cudaMallocHost(&(this->buffer_), size_);
    cudaErrorCheck(err, "Impossible to allocate Pinned host memory.");
}

void Buffer::destroy(CUDA& device, Stream& stream) {
    auto err = cudaFreeAsync(this->buffer_, stream.getStream(device));
    cudaErrorCheck(err, "Impossible to release memory.");
    this->buffer_ = nullptr;
    this->size_ = 0;
    this->device_ = std::make_shared<UnknownDevice>();
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

}  // namespace kmm
