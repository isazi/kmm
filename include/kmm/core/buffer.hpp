#pragma once

#include <iosfwd>
#include <memory>
#include <vector>

#include "kmm/core/data_type.hpp"
#include "kmm/core/identifiers.hpp"

namespace kmm {

/**
 * Represents the layout of a buffer, including its size and alignment.
 */
struct DataLayout {
    DataLayout repeat(size_t n) {
        size_t remainder = size_in_bytes % alignment;
        size_t padding = remainder != 0 ? alignment - remainder : 0;
        return {(size_in_bytes + padding) * n, alignment};
    }

    template<typename T>
    static DataLayout for_type() {
        return DataLayout {sizeof(T), alignof(T)};
    }

    static DataLayout for_type(DataType dtype) {
        return DataLayout {dtype.size_in_bytes(), dtype.alignment()};
    }

    size_t size_in_bytes;
    size_t alignment;
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
    MemoryId memory_id;
    DataLayout layout;
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

inline std::ostream& operator<<(std::ostream& f, AccessMode mode) {
    switch (mode) {
        case AccessMode::Read:
            return f << "Read";
        case AccessMode::ReadWrite:
            return f << "ReadWrite";
        case AccessMode::Exclusive:
            return f << "Exclusive";
    }

    return f;
}

}  // namespace kmm

template<>
struct fmt::formatter<kmm::AccessMode>: fmt::ostream_formatter {};