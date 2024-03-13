#pragma once

#include "kmm/identifiers.hpp"
#include "kmm/runtime.hpp"
#include "kmm/utils/macros.hpp"

namespace kmm {

class RuntimeHandle;
class Block;

class ArrayBase {
  public:
    ArrayBase(std::shared_ptr<Block> block = nullptr) : m_block(std::move(block)) {}
    virtual ~ArrayBase() = default;

    virtual size_t rank() const = 0;
    virtual index_t size(size_t axis) const = 0;

    /**
     * Returns the `Block` that is associated with this array.
     */
    std::shared_ptr<Block> block() const;

    /**
     * Returns `true` if this array has a block assigned to it, and `false` otherwise.
     */
    bool has_block() const;

    /**
     * Returns the id of the assigned block.
     */
    BlockId id() const;

    /**
     * Returns the `Runtime` associated with this array.
     */
    RuntimeHandle runtime() const;

    /**
     * Block the current thread until all operations on this array have completed.
     */
    void synchronize() const;

    /**
     * Prefetch this array in the provided memory. This is useful for optimization purposes, but
     * not necessary for correctness.
     */
    EventId prefetch(MemoryId memory_id, EventList dependencies = {}) const;

    /**
     * Read the content of this array into the given pointer `dst_ptr` that has a size of `nbytes`
     * bytes.
     */
    void read_bytes(void* dst_ptr, size_t nbytes) const;

    /**
     * Returns the total volume of this array (i.e., the total number of elements).
     *
     * @return The volume of this array.
     */
    index_t size() const;

    /**
     * Check if the array contains no elements (i.e., if `size() == 0`)
     *
     * @return `true` if `size() == 0`, and `false` otherwise.
     */
    bool is_empty() const;

  protected:
    std::shared_ptr<Block> m_block;
};

/**
 * Stores the `BlockId` of an array and deletes the array in its destructor.
 */
class Block {
    KMM_NOT_COPYABLE_OR_MOVABLE(Block)

  public:
    Block(std::shared_ptr<Runtime> runtime, BlockId id);
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
    Runtime& runtime() const {
        return *m_runtime;
    }

  private:
    BlockId m_id;
    std::shared_ptr<Runtime> m_runtime;
};

}  // namespace kmm