#pragma once

#include <array>
#include <utility>

#include "runtime.hpp"
#include "types.hpp"

namespace kmm {

class Buffer {
  public:
    Buffer(const Runtime& rt, BufferLayout layout) : runtime_(rt), layout_(layout) {
        id_ = rt.create_buffer(layout);
    }

    Buffer(const Buffer& that) : id_(that.id_), layout_(that.layout_), runtime_(that.runtime_) {
        runtime_.increment_buffer_refcount(id_);
    }

    Buffer& operator=(const Buffer& that) {
        if (id_ != that.id_ || runtime_ != that.runtime_) {
            BufferId old_id = std::exchange(id_, INVALID_BUFFER_ID);
            this->runtime_.decrement_buffer_refcount(old_id);
            that.runtime_.increment_buffer_refcount(that.id_);
        }

        id_ = that.id_;
        runtime_ = that.runtime_;
        layout_ = that.layout_;
    }

    ~Buffer() {
        runtime_.decrement_buffer_refcount(id_);
    }

    BufferId id() const {
        return id_;
    }

    const Runtime& runtime() const {
        return runtime_;
    }

    BufferLayout layout() const {
        return layout_;
    }

  private:
    BufferId id_;
    BufferLayout layout_;
    Runtime runtime_;
};

template<typename T, size_t N>
class Array {
    Array(const Runtime& rt, std::array<index_t, N> shape) : shape_(shape) {
        auto layout = BufferLayout {
            .alignment = alignof(T),
            .size_in_bytes = sizeof(T) * size(),
        };

        buffer_ = Buffer(rt, layout);
    }

    const Buffer& buffer() const {
        return buffer_;
    }

    BufferId id() const {
        return buffer_.id();
    }

    const Runtime& runtime() const {
        return buffer_.runtime();
    }

    std::array<index_t, N> shape() const {
        return shape_;
    }

    index_t size() const {
        index_t total = 1;
        for (size_t i = 0; i < N; i++) {
            total *= shape[i];
        }

        return total;
    }

    index_t size(size_t axis) const {
        return shape_[axis];
    }

  private:
    Buffer buffer_;
    std::array<index_t, N> shape_;
};

}  // namespace kmm