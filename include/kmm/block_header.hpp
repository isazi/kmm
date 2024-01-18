#pragma once

#include <string>

#include "kmm/utils/waker.hpp"

namespace kmm {

struct BlockLayout {
    size_t num_bytes;
    size_t alignment;
};

class BlockHeader {
  public:
    virtual ~BlockHeader() = default;
    virtual BlockLayout layout() const = 0;
    virtual std::string name() const = 0;
};

class ArrayHeader final: public BlockHeader {
  private:
    ArrayHeader(
        index_t length,
        size_t element_size,
        size_t element_align,
        const std::type_info& element_type) :
        length_(length),
        element_size_(element_size),
        element_align_(element_align),
        element_type_(element_type) {}

  public:
    template<typename T>
    static ArrayHeader for_type(index_t num_elements) {
        return {num_elements, sizeof(T), alignof(T), typeid(T)};
    }

    BlockLayout element_layout() const {
        return {element_size_, element_align_};
    }

    const std::type_info& element_type() const {
        return element_type_;
    }

    index_t num_elements() const {
        return length_;
    }

    BlockLayout layout() const final {
        return {
            element_size_ * length_,
            element_align_,
        };
    }

    std::string name() const final {
        return element_type().name();
    }

  private:
    index_t length_;
    size_t element_size_;
    size_t element_align_;
    const std::type_info& element_type_;
};

template<typename T>
class ScalarHeader final: public BlockHeader {
  public:
    ScalarHeader(T value = {}) : m_value(std::move(value)) {}

    std::string name() const override {
        return typeid(T).name();
    }

    BlockLayout layout() const override {
        return BlockLayout {
            .num_bytes = 0,
            .alignment = 1,
        };
    }

    T& get() {
        return m_value;
    }

    const T& get() const {
        return m_value;
    }

  private:
    T m_value;
};

}  // namespace kmm