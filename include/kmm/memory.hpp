#pragma once

#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>

#include "kmm/result.hpp"
#include "kmm/types.hpp"

namespace kmm {

class MemoryAllocation {
  public:
    virtual ~MemoryAllocation() = default;
};

class IMemoryCompletion {
  public:
    virtual ~IMemoryCompletion() = default;
    virtual void complete(Result<void>) = 0;
};

class MemoryCompletion {
  public:
    explicit MemoryCompletion(std::shared_ptr<IMemoryCompletion> c);
    ~MemoryCompletion();

    MemoryCompletion(MemoryCompletion&&) noexcept = default;
    MemoryCompletion& operator=(MemoryCompletion&&) noexcept = default;

    MemoryCompletion(const MemoryCompletion&) = delete;
    MemoryCompletion& operator=(const MemoryCompletion&) = delete;

    void complete(Result<void> = {});

  public:
    std::shared_ptr<IMemoryCompletion> m_completion;
};

class Memory {
  public:
    virtual ~Memory() = default;

    /**
     * @brief Allocates memory of a specified size.
     * @param id The identifier for the memory.
     * @param num_bytes The number of bytes to allocate.
     * @return An optional unique pointer to a MemoryAllocation object on success,
     *         or an empty optional if allocation fails.
     */
    virtual std::optional<std::unique_ptr<MemoryAllocation>> allocate(
        MemoryId id,
        size_t num_bytes) = 0;

    /**
     * @brief Deallocates the memory identified by the given id.
     * @param id The identifier of the memory to deallocate.
     * @param allocation A unique pointer to the MemoryAllocation object to be deallocated.
     */
    virtual void deallocate(MemoryId id, std::unique_ptr<MemoryAllocation> allocation) = 0;

    /**
     * @brief Asynchronously copies memory from source to destination.
     * @param completion Callback function to be called upon completion.
     */
    virtual void copy_async(
        MemoryId src_id,
        const MemoryAllocation* src_alloc,
        size_t src_offset,
        MemoryId dst_id,
        const MemoryAllocation* dst_alloc,
        size_t dst_offset,
        size_t num_bytes,
        MemoryCompletion completion) = 0;

    /**
     * @brief Checks if a copy operation is possible between the given source and destination.
     * @return True if the copy operation is possible, false otherwise.
     */
    virtual bool is_copy_possible(MemoryId src_id, MemoryId dst_id) = 0;

    /**
     * @brief Asynchronously fills memory with specified bytes.
     * @param fill_bytes The byte pattern to fill the memory with. The pattern will be repeated
     *                   if the buffer is larger than the byte pattern.
     * @param completion Callback function to be called upon completion.
     */
    virtual void fill_async(
        MemoryId dst_id,
        const MemoryAllocation* dst_alloc,
        size_t dst_offset,
        size_t num_bytes,
        std::vector<uint8_t> fill_bytes,
        MemoryCompletion completion) = 0;
};

}  // namespace kmm