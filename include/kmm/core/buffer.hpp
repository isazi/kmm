#pragma once

#include <memory>

#include "identifiers.hpp"

namespace kmm {

/**
 * Represents the layout of a buffer, including its size and alignment.
 */
struct BufferLayout {
    size_t size_in_bytes;
    size_t alignment;

    template<typename T>
    static BufferLayout for_type(size_t n = 1) {
        return {n * sizeof(T), alignof(T)};
    }
};

/**
 * This enum is used to specify how a buffer can be accessed: read-only, read-write, or exclusive.
 */
enum struct AccessMode {
    Read,  ///< Read-only access to the buffer.
    ReadWrite,  ///< Read and write access to the buffer.
    Exclusive  ///< Exclusive access, implying full control over the buffer.
};

/**
 *  Represents the requirements for accessing a buffer.
 */
struct BufferRequirement {
    BufferId buffer_id;
    MemoryId memory_id;
    AccessMode access_mode;
};

/**
 * Provides access to a buffer with specific properties.
 */
struct BufferAccessor {
    BufferId buffer_id;
    MemoryId memory_id;
    BufferLayout layout;
    bool is_writable;
    void* address;
};

/**
 * Manages the lifetime and access of a buffer. The `BufferGuard` struct extends `BufferAccessor`
 * and ensures that the buffer's access and lifetime are properly managed using a shared tracker.
 */
struct BufferGuard: public BufferAccessor {
    BufferGuard(BufferAccessor accessor, std::shared_ptr<void> tracker) :
        BufferAccessor(accessor),
        m_tracker(tracker) {}

    BufferAccessor accessor() const {
        return *this;
    }

    std::shared_ptr<void> m_tracker;
};

}  // namespace kmm