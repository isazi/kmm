#pragma once

#include "kmm/runtime.hpp"

namespace kmm {

/**
 * Represents a memory buffer within the runtime system. It provides functionalities to prefetch
 * data, synchronize operations, and manage the memory block lifecycle.
 */
class Block {
    KMM_NOT_COPYABLE_OR_MOVABLE(Block)

  public:
    Block(std::shared_ptr<RuntimeImpl> runtime, BlockId id = BlockId::invalid());
    ~Block();

    /**
     * Returns the unique identifier of this buffer.
     */
    BlockId id() const {
        return m_id;
    }

    /**
     * Returns the runtime associated with this buffer.
     */
    Runtime runtime() const {
        return m_runtime;
    }

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
     * Delete this buffer. It is not necessary to call this method manually, since it will also be
     * called by the destructor.
     */
    void destroy();

    /**
     *
     */
    BlockId release();

  private:
    BlockId m_id = BlockId::invalid();
    std::shared_ptr<RuntimeImpl> m_runtime;
};


}