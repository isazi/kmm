#pragma once

#include <string>

#include "kmm/types.hpp"

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

class ArrayHeader: public BlockHeader {
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
        auto element = element_layout();

        return BlockLayout {
            .num_bytes = element.num_bytes * length_,
            .alignment = element.alignment,
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
class Scalar;

class ScalarHeader: public BlockHeader {
  public:
    virtual const std::type_info& type() const = 0;

    BlockLayout layout() const override {
        return BlockLayout {
            .num_bytes = 0,
            .alignment = 1,
        };
    }

    bool is(const std::type_info& t) const {
        return type() == t;
    }

    template<typename T>
    bool is() const {
        return is(typeid(T));
    }

    template<typename T>
    T* get_if() {
        if (auto ptr = dynamic_cast<Scalar<T>*>(this)) {
            return ptr->get();
        } else {
            return nullptr;
        }
    }

    template<typename T>
    const T* get_if() const {
        if (auto ptr = dynamic_cast<const Scalar<T>*>(this)) {
            return ptr->get();
        } else {
            return nullptr;
        }
    }
};

template<typename T>
class Scalar: public ScalarHeader {
  public:
    Scalar() {}
    Scalar(T value) : m_value(std::move(value)) {}

    const std::type_info& type() const override {
        return typeid(T);
    }

    void set(T new_value) {
        m_value = std::move(new_value);
    }

    T& get() {
        return m_value;
    }

    const T& get() const {
        return m_value;
    }

    operator T() const {
        return get();
    }

  private:
    T m_value = {};
};

}  // namespace kmm