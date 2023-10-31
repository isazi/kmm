#pragma once

#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>

#include "kmm/types.hpp"

namespace kmm {

class Allocation {
  public:
    virtual ~Allocation() = default;
};

class Completion {
  public:
    virtual ~Completion() = default;
    virtual void complete() = 0;
};

class Memory {
  public:
    virtual ~Memory() = default;
    virtual std::optional<std::unique_ptr<Allocation>> allocate(
        kmm::DeviceId id,
        size_t num_bytes) = 0;
    virtual void deallocate(kmm::DeviceId id, std::unique_ptr<Allocation> allocation) = 0;

    virtual void copy_async(
        kmm::DeviceId src_id,
        const Allocation* src_alloc,
        kmm::DeviceId dst_id,
        const Allocation* dst_alloc,
        size_t num_bytes,
        std::unique_ptr<Completion> completion) = 0;

    virtual bool is_copy_possible(kmm::DeviceId src_id, kmm::DeviceId dst_id) = 0;
};

}