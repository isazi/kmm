#pragma once

#include <memory>
#include <optional>

#include "fmt/format.h"

#include "kmm/event_list.hpp"
#include "kmm/identifiers.hpp"
#include "kmm/utils/completion.hpp"
#include "kmm/utils/result.hpp"

namespace kmm {

class MemoryAllocation {
  public:
    virtual ~MemoryAllocation() = default;

    /**
     * Copy from host memory to this allocation.
     *
     * @param dst_addr The host memory source address.
     * @param dst_offset The offset within the destination memory.
     * @param num_bytes The number of bytes to copy.
     */
    virtual void copy_from_host_sync(const void* src_addr, size_t dst_offset, size_t num_bytes) = 0;

    /**
     * Copy from this memory allocation to host memory.
     *
     * @param src_offset The offset within the this allocation.
     * @param dst_addr The host memory destination address.
     * @param num_bytes The number of bytes to copy.
     */
    virtual void copy_to_host_sync(size_t src_offset, void* dst_addr, size_t num_bytes) const = 0;
};

class Memory {
  public:
    virtual ~Memory() = default;

    /**
     * @brief Allocates memory of a specified size.
     * @param id The identifier for the memory.
     * @param num_bytes The number of bytes to allocate.
     * @return An `MemoryAllocation` object on success, or an empty optional if allocation fails.
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
     * @brief Asynchronously copies data from source to destination.
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
        Completion completion) = 0;

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
        std::vector<uint8_t> fill_pattern,
        Completion completion) = 0;
};

}  // namespace kmm