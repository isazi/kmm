#pragma once

namespace kmm {

enum struct AccessMode { Read, ReadWrite, Exclusive };

struct BufferLayout {
    size_t size_in_bytes;
    size_t alignment;

    template<typename T>
    static BufferLayout for_type(size_t n = 1) {
        return {n * sizeof(T), alignof(T)};
    }
};

struct BufferAccessor {
    BufferId buffer_id;
    MemoryId memory_id;
    BufferLayout layout;
    bool is_writable;
    void* address;
};

}  // namespace kmm