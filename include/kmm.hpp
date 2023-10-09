#pragma once

#include <cstddef>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace kmm {

// Compute devices
class DeviceType {
  public:
    DeviceType() = default;
    DeviceType(const DeviceType&) = delete;

  public:
    virtual ~DeviceType() = default;
};

class UnknownDevice: public DeviceType {};

class CPU: public DeviceType {};

class GPU: public DeviceType {
  public:
    GPU();
    GPU(unsigned int device_id);
    virtual ~GPU() = default;
    unsigned int device_id;
};

class CUDA: public GPU {
    using GPU::GPU;
};

// Memories
class MemoryType {
  public:
    MemoryType() = default;
    MemoryType(const MemoryType&) = delete;

  public:
    virtual ~MemoryType() = default;
};

class DefaultMemory: public MemoryType {};

class CUDAPinned: public DefaultMemory {};

// Pointer

template<typename Type>
class Pointer {
  public:
    Pointer();
    Pointer(unsigned int id);

    const std::type_info& type() const {
        return typeid(Type);
    }

    unsigned int id() const {
        return id_;
    }

  protected:
    unsigned int id_;
};

// Stream

class Stream;
#ifdef USE_CUDA
    #include "kmm.cuh"
#endif

struct Allocation {};
struct Executor {};
struct ExecutorContext {};

enum struct AccessMode {
    READ,
    WRITE,
};

struct BufferRequirement {
    unsigned int buffer_id;
    unsigned int memory_id;
    AccessMode mode;
};

struct BufferAccess {
    unsigned int buffer_id;
    unsigned int memory_id;
    AccessMode mode;
    std::shared_ptr<Allocation> allocation;
};

// Task

class Task {
  public:
    virtual void execute(std::vector<BufferAccess> buffers) const = 0;
    virtual ~Task() = default;
};

template<typename D, typename F, typename... Args>
class TaskImpl: Task {
    TaskImpl(D device, F fun, Args... args) :
        device_(device),
        fun_(std::move(fun)),
        args_(std::move(args)...) {}

    void execute(std::vector<BufferAccess> buffers) const override {
        execute_with_indices(buffers, std::make_index_sequence<sizeof...(Args)>());
    }

  private:
    template<size_t... Is>
    void execute(std::vector<BufferAccess> buffers, std::index_sequence<Is...>) const override {}

    D device_;
    F fun_;
    std::tuple<Args...> args_;
};

template<typename T>
struct TaskArgument {
    using type = T;

    static type call(T input, std::vector<BufferRequirement>& reqs) {
        return input;
    }
};

template<typename T>
struct TaskArgument<Pointer<T>> {
    using type = Pointer<T>;

    static type call(Pointer<T> input, std::vector<BufferRequirement>& reqs) {
        reqs.push_back(BufferRequirement {
            .buffer_id = input.id(),
            .memory_id = 0,
            .mode = AccessMode::READ,
        });

        return input;
    }
};

// Manager

class ManagerImpl;

class Manager {
  public:
    Manager();

    void submit_task(
        unsigned int executor_id,
        std::shared_ptr<Task> task,
        std::vector<BufferRequirement> reqs) const;

    template<typename D, typename F, typename... Args>
    void submit(D device, F fun, Args&&... args) const {
        unsigned int executor_id = device.select_executor(*this);

        std::vector<BufferRequirement> reqs;
        auto task = std::make_shared<TaskImpl<
            std::decay_t<D>,
            std::decay_t<F>,
            typename TaskArgument<std::decay_t<Args>>::type...>>(
            std::move(device),
            std::move(fun),
            TaskArgument<std::decay_t<Args>>::call(std::forward<Args>(args), reqs)...);

        return submit_task(executor_id, std::move(task), std::move(reqs));
    }

    const std::vector<std::shared_ptr<MemoryType>>& memories() const;
    const std::vector<std::shared_ptr<Executor>>& executors() const;

    template<typename Type>
    Pointer<Type> create(std::size_t size) {
        return Pointer<Type>(this->create_impl(size));
    }

    template<typename Type>
    Pointer<Type> create(std::size_t size, CUDAPinned& memory) {
        return Pointer<Type>(this->create_impl(size, memory));
    }

    template<typename Type>
    void move_to(CPU& device, const Pointer<Type>& pointer) {
        this->move_to_impl(device, pointer.id());
    }

    template<typename Type>
    void move_to(CUDA& device, const Pointer<Type>& pointer) {
        this->move_to_impl(device, pointer.id());
    }

    template<typename Type>
    void release(const Pointer<Type>& pointer) {
        release_impl(pointer.id());
    }

    template<typename Type>
    void copy_release(const Pointer<Type>& pointer, void* target) {
        this->copy_release_impl(pointer.id(), target);
    }

  private:
    unsigned int create_impl(size_t size);
    unsigned int create_impl(size_t size, CUDAPinned& memory);
    void move_to_impl(CPU& device, unsigned int pointer_id);
    void move_to_impl(CUDA& device, unsigned int pointer_id);
    void release_impl(unsigned int pointer_id);
    void copy_release_impl(unsigned int pointer_id, void* target);
    std::shared_ptr<ManagerImpl> impl_;
};

// Buffer

class Buffer {
  public:
    Buffer();
    Buffer(std::size_t size);
    Buffer(std::size_t size, CUDAPinned& memory);
    Buffer(CPU& device, std::size_t size);
    Buffer(CUDA& device, std::size_t size);
    // Manipulate size
    std::size_t size() const;
    void set_size(std::size_t size);
    // Manipulate device
    std::shared_ptr<DeviceType> device();
    void set_device(CPU& device);
    void set_device(CUDA& device);
    // Manipulate special memory
    std::shared_ptr<MemoryType> memory();
    void set_memory(CUDAPinned& memory);
    // Return true if the buffer is allocated
    bool is_allocated() const;
    // Allocate memory
    void allocate();
    void allocate(CUDAPinned& memory);
    void allocate(CUDA& device, Stream& stream);
    // Free memory
    void destroy();
    void destroy(CUDA& device, Stream& stream);
    // Return a pointer to the allocated buffer
    void* getPointer();

  private:
    void* buffer_;
    std::size_t size_;
    std::shared_ptr<DeviceType> device_;
    std::shared_ptr<MemoryType> memory_;
};

// Misc

inline bool is_cpu(DeviceType& device) {
    return dynamic_cast<const CPU*>(&device) != nullptr;
}

inline bool is_cuda_pinned(Buffer& buffer) {
    return dynamic_cast<CUDAPinned*>(buffer.memory().get()) != nullptr;
}

inline bool is_pinned(Buffer& buffer) {
    return is_cuda_pinned(buffer);
}

inline bool on_cpu(Buffer& buffer) {
    return dynamic_cast<CPU*>(buffer.device().get()) != nullptr;
}

inline bool on_cuda(Buffer& buffer) {
    return dynamic_cast<CUDA*>(buffer.device().get()) != nullptr;
}

inline bool same_device(const DeviceType& a, const DeviceType& b) {
    if (dynamic_cast<const CPU*>(&a) != nullptr) {
        return dynamic_cast<const CPU*>(&b) != nullptr;
    }

    if (auto a_ptr = dynamic_cast<const GPU*>(&a)) {
        if (auto b_ptr = dynamic_cast<const GPU*>(&b)) {
            return a_ptr->device_id == b_ptr->device_id;
        }
    }

    return false;
}

void cudaCopyD2H(CUDA& device, Buffer& source, Buffer& target, Stream& stream);
void cudaCopyD2H(CUDA& device, Buffer& source, void* target, Stream& stream);
void cudaCopyH2D(CUDA& device, Buffer& source, Buffer& target, Stream& stream);
void cudaCopyD2D(CUDA& device, Buffer& source, Buffer& target, Stream& stream);

// Manager

// Pointer

template<typename Type>
Pointer<Type>::Pointer() {}

template<typename Type>
Pointer<Type>::Pointer(unsigned int id) {
    this->id_ = id;
}

// WritePointer

template<typename P>
class WritePointer {
  public:
    WritePointer(P& inner) : inner_(inner) {}

    P& get() const {
        return inner_;
    }

  private:
    P& inner_;
};

template<typename P>
WritePointer<P> write(P& ptr) {
    return ptr;
}

}  // namespace kmm
