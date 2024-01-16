#pragma once

#include "kmm/identifiers.hpp"
#include "kmm/runtime_impl.hpp"
#include "kmm/utils.hpp"

namespace kmm {

class Runtime;

/**
 * Represents a memory buffer within the runtime system. It provides functionalities to prefetch
 * data, synchronize operations, and manage the memory block lifecycle.
 */
class Block {
    KMM_NOT_COPYABLE_OR_MOVABLE(Block)

  public:
    Block(std::shared_ptr<RuntimeImpl> runtime, BlockId id);
    ~Block();

    static std::shared_ptr<Block> create(
        std::shared_ptr<RuntimeImpl> runtime,
        std::unique_ptr<BlockHeader> header,
        const void* data_ptr,
        size_t num_bytes,
        MemoryId memory_id = MemoryId(0));

    static std::shared_ptr<Block> create(
        std::shared_ptr<RuntimeImpl> runtime,
        std::unique_ptr<BlockHeader> header) {
        return create(std::move(runtime), std::move(header), nullptr, 0);
    }

    /**
     * Returns the unique identifier of this buffer.
     */
    BlockId id() const {
        return m_id;
    }

    /**
     * Returns the runtime associated with this buffer.
     */
    Runtime runtime() const;

    /**
     * Prefetch this buffer in the provided memory.
     *
     * @param memory_id The identifier of the memory.
     * @param dependencies Events that should complete before the prefetch occurs.
     * @return The identifier of the prefetch event.
     */
    EventId prefetch(MemoryId memory_id, EventList dependencies = {}) const;

    /**
     * Submit a barrier the runtime system. The barrier completes once all the events submitted
     * to the runtime system so far associated with this buffer have finished execution.
     *
     * @return The identifier of the barrier.
     */
    EventId submit_barrier() const;

    /**
     * Blocks until all the events associated with this buffer have finished execution.
     */
    void synchronize() const;

    /**
     * Release this block.
     */
    BlockId release();

    /**
     * Fetch the header of this block.
     *
     * @return The block header.
     */
    std::shared_ptr<BlockHeader> header() const;

    /**
     * Read the content of this block.
     *
     * @return The block content.
     */
    std::shared_ptr<BlockHeader> read(void* dst_ptr, size_t nbytes) const;

  private:
    BlockId m_id;
    std::shared_ptr<RuntimeImpl> m_runtime;
};

}  // namespace kmm