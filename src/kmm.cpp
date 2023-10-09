#include "kmm.hpp"

#include <cstring>

namespace kmm {

// Manager

struct ManagerImpl {
    unsigned int next_allocation = 0;
    unsigned int next_task = 0;
    std::map<unsigned int, Stream> streams;
    std::map<unsigned int, Buffer> allocations;
    std::map<unsigned int, Task> tasks;
    // Check if a device has a stream allocated
    bool stream_exist(unsigned int stream);
};

Manager::Manager() {
    impl_ = std::make_shared<ManagerImpl>();
}

unsigned int Manager::create_impl(size_t size) {
    unsigned int allocation_id;
    allocation_id = impl_->next_allocation++;
    impl_->allocations[allocation_id] = Buffer(size);
    return allocation_id;
}

unsigned int Manager::create_impl(size_t size, CUDAPinned& memory) {
    unsigned int allocation_id;
    allocation_id = impl_->next_allocation++;
    impl_->allocations[allocation_id] = Buffer(size, memory);
    return allocation_id;
}

bool ManagerImpl::stream_exist(unsigned int stream) {
    return this->streams.find(stream) != this->streams.end();
}

Task Manager::run() {
    unsigned int task_id = impl_->next_task++;
    impl_->tasks[task_id] = Task(task_id);
    return impl_->tasks[task_id];
}

void Manager::move_to_impl(CPU& device, unsigned int pointer_id) {
    auto source_buffer = impl_->allocations[pointer_id];

    if (on_cpu(source_buffer)) {
        return;
    }
    if (source_buffer.is_allocated()) {
        auto source_device = *(dynamic_cast<CUDA*>(source_buffer.getDevice().get()));
        auto target_buffer = Buffer(device, source_buffer.size());
        auto stream = impl_->streams[source_device.device_id];
        if (is_cuda_pinned(source_buffer)) {
            target_buffer.allocate(*(dynamic_cast<CUDAPinned*>(source_buffer.getMemory().get())));
        } else {
            target_buffer.allocate();
        }
        cudaCopyD2H(source_device, source_buffer, target_buffer, stream);
        source_buffer.destroy(source_device, stream);
        impl_->allocations[pointer_id] = target_buffer;
    } else {
        source_buffer.setDevice(device);
    }
}
void Manager::move_to_impl(CUDA& device, unsigned int pointer_id) {
    auto source_buffer = impl_->allocations[pointer_id];

    if (!source_buffer.is_allocated()) {
        source_buffer.setDevice(device);
        return;
    }
    auto target_buffer = Buffer(device, source_buffer.size());
    if (on_cuda(source_buffer)) {
        // source_buffer is allocated on a CUDA GPU
        auto source_device = *(dynamic_cast<CUDA*>(source_buffer.getDevice().get()));
        if (same_device(device, source_device)) {
            return;
        }
        auto stream = impl_->streams[source_device.device_id];
        target_buffer.allocate(device, stream);
        cudaCopyD2D(device, source_buffer, target_buffer, stream);
        source_buffer.destroy(source_device, stream);
    } else if (on_cpu(source_buffer)) {
        // source_buffer is allocated on a CPU
        auto stream = impl_->streams[device.device_id];
        target_buffer.allocate(device, stream);
        if (is_cuda_pinned(source_buffer)) {
            target_buffer.setMemory(*(dynamic_cast<CUDAPinned*>(source_buffer.getMemory().get())));
        }
        cudaCopyH2D(device, source_buffer, target_buffer, stream);
        source_buffer.destroy();
    }
    impl_->allocations[pointer_id] = target_buffer;
}

void Manager::release_impl(unsigned int pointer_id) {
    auto buffer = impl_->allocations[pointer_id];

    if (on_cpu(buffer)) {
        buffer.destroy();
    } else if (on_cuda(buffer)) {
        auto device = *(dynamic_cast<CUDA*>(buffer.getDevice().get()));
        auto stream = impl_->streams[device.device_id];
        buffer.destroy(device, stream);
    }
}

void Manager::copy_release_impl(unsigned int pointer_id, void* target) {
    auto buffer = impl_->allocations[pointer_id];

    if (on_cpu(buffer)) {
        ::memcpy(target, buffer.getPointer(), buffer.size());
    } else if (on_cuda(buffer)) {
        auto device = *(dynamic_cast<CUDA*>(buffer.getDevice().get()));
        auto stream = impl_->streams[device.device_id];
        cudaCopyD2H(device, buffer, target, stream);
    }

    this->release_impl(pointer_id);
}

// Buffer

Buffer::Buffer() {
    this->buffer_ = nullptr;
    this->size_ = 0;
    this->device_ = std::make_shared<UnknownDevice>();
    this->memory_ = std::make_shared<DefaultMemory>();
}

Buffer::Buffer(std::size_t size) {
    this->buffer_ = nullptr;
    this->size_ = size;
    this->device_ = std::make_shared<UnknownDevice>();
    this->memory_ = std::make_shared<DefaultMemory>();
}

Buffer::Buffer(std::size_t size, CUDAPinned& memory) {
    this->buffer_ = nullptr;
    this->size_ = size;
    this->device_ = std::make_shared<CPU>();
    this->memory_ = std::make_shared<CUDAPinned>();
}

Buffer::Buffer(CPU& device, std::size_t size) {
    this->buffer_ = nullptr;
    this->size_ = size;
    this->device_ = std::make_shared<CPU>();
    this->memory_ = std::make_shared<DefaultMemory>();
}

Buffer::Buffer(CUDA& device, std::size_t size) {
    this->buffer_ = nullptr;
    this->size_ = size;
    this->device_ = std::make_shared<CUDA>(device.device_id);
    this->memory_ = std::make_shared<DefaultMemory>();
}

std::size_t Buffer::size() const {
    return this->size_;
}

void Buffer::setSize(std::size_t size) {
    this->size_ = size;
}

std::shared_ptr<DeviceType> Buffer::getDevice() {
    return this->device_;
}

void Buffer::setDevice(CPU& device) {
    this->device_ = std::make_shared<CPU>();
}

void Buffer::setDevice(CUDA& device) {
    this->device_ = std::make_shared<CUDA>(device.device_id);
}

std::shared_ptr<MemoryType> Buffer::getMemory() {
    return this->memory_;
}

void Buffer::setMemory(CUDAPinned& memory) {
    this->memory_ = std::make_shared<CUDAPinned>();
}

bool Buffer::is_allocated() const {
    return this->buffer_ != nullptr;
}

void Buffer::allocate() {
    this->buffer_ = malloc(this->size_);
}

void Buffer::destroy() {
    free(this->buffer_);
    this->buffer_ = nullptr;
    this->size_ = 0;
    this->device_ = std::make_shared<UnknownDevice>();
}

void* Buffer::getPointer() {
    return this->buffer_;
}

// GPU

GPU::GPU() {
    this->device_id = 0;
}

GPU::GPU(unsigned int device_id) {
    this->device_id = device_id;
}

// Task

Task::Task() {
    this->id = 0;
}

Task::Task(unsigned int id) {
    this->id = id;
}

}  // namespace kmm
