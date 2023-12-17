#pragma once

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

class ArrayBaseHeader: public BlockHeader {
  public:
    ArrayBaseHeader(index_t n) : length_(n) {}

    virtual BlockLayout element_layout() const = 0;
    virtual const std::type_info& element_type() const = 0;

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
};

template<typename T>
class ArrayHeader: public ArrayBaseHeader {
  public:
    ArrayHeader(index_t n = 0) : ArrayBaseHeader(n) {}

    BlockLayout element_layout() const final {
        return BlockLayout {.num_bytes = sizeof(T), .alignment = alignof(T)};
    }

    const std::type_info& element_type() const final {
        return typeid(T);
    }
};

template<typename T>
class Scalar;

class ScalarBase: public BlockHeader {
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
class Scalar: public ScalarBase {
  public:
    Scalar() {}
    Scalar(T value) : m_value(std::move(value)) {}

    const std::type_info& type() const override {
        return typeid(T);
    }

    T& get() {
        return m_value;
    }

    const T& get() const {
        return m_value;
    }

    operator T&() {
        return get();
    }

    operator const T&() const {
        return get();
    }

    operator T() const {
        return get();
    }

  private:
    T m_value = {};
};

}  // namespace kmm