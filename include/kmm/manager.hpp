#pragma once

#include "task.hpp"
#include "types.hpp"

namespace kmm {

// Memories
class MemoryType {
  public:
    virtual ~MemoryType() = default;
    virtual std::string name() const = 0;
    virtual std::optional<ExecutorId> executor_affinity() const {
        return {};
    }

    virtual std::optional<std::shared_ptr<Allocation>>
    allocate_buffer(size_t nbytes, size_t alignment) const = 0;
    virtual void release_buffer(std::shared_ptr<Allocation> alloc) const = 0;
};

struct BufferRequirement {
    BufferId buffer_id;
    MemoryId memory_id;
    AccessMode mode;
};

template<typename Arg>
class TaskArgumentPack {
  public:
    using type = Arg;

    static type call(Arg input, std::vector<BufferRequirement>& reqs) {
        return input;
    }
};

class ManagerImpl;

class Manager {
  public:
    Manager();

    void submit_task(
        ExecutorId executor_id,
        std::shared_ptr<Task> task,
        std::vector<BufferRequirement> reqs) const;

    template<typename D, typename F, typename... Args>
    void submit(D device, F fun, Args&&... args) const {
        ExecutorId executor_id = device.select_executor(*this);

        std::vector<BufferRequirement> reqs;
        auto task = std::make_shared<TaskImpl<
            D::memory_space,
            std::decay_t<F>,
            typename TaskArgumentPack<std::decay_t<Args>>::type...>>(
            std::move(device),
            std::move(fun),
            TaskArgumentPack<std::decay_t<Args>>::call(std::forward<Args>(args), reqs)...);

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

    template<typename Device>
    void move_to(Device&& device, const Pointer<Type>& pointer) {
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
    BufferId create_impl(size_t size);
    BufferId create_impl(size_t size, CUDAPinned& memory);
    void move_to_impl(MemoryId memory_id, BufferId pointer_id);
    void release_impl(BufferId pointer_id);
    void copy_release_impl(BufferId pointer_id, void* target);
    std::shared_ptr<ManagerImpl> impl_;
};

}  // namespace kmm