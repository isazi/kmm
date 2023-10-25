#pragma once

#include <string>

namespace kmm {

template<typename Tag, typename T>
class IntegerType {
  public:
    explicit IntegerType(T value) : m_value(value) {}

    static constexpr IntegerType invalid() {
        return IntegerType(std::numeric_limits<T>::max());
    }

    T get() const {
        return m_value;
    }

    operator T() const {
        return get();
    }

    IntegerType& operator=(const T& v) {
        m_value = v;
        return *this;
    }

    bool operator==(const IntegerType& that) const {
        return m_value == that.m_value;
    }

    bool operator<(const IntegerType& that) const {
        return m_value < that.m_value;
    }

    bool operator!=(const IntegerType& that) const {
        return !(*this == that);
    }

    bool operator<=(const IntegerType& that) const {
        return !(this > that);
    }

    bool operator>(const IntegerType& that) const {
        return that < *this;
    }

    bool operator>=(const IntegerType& that) const {
        return that <= *this;
    }

  private:
    T m_value;
};

using DeviceId = IntegerType<struct DeviceTag, uint8_t>;
using TaskId = IntegerType<struct TaskTag, uint64_t>;
using VirtualBufferId = IntegerType<struct VirtualBufferTag, uint64_t>;
using BufferId = IntegerType<struct BufferTag, uint64_t>;

struct BufferDescription {
    size_t num_bytes;
    size_t alignment;
    DeviceId home;
    std::string name;
};

enum class AccessMode {
    Read,
    Write,
};

struct VirtualBufferRequirement {
    VirtualBufferId buffer_id;
    AccessMode mode;
};

class Runtime;
class RuntimeImpl;

}  // namespace kmm

namespace std {
template<typename Tag, typename T>
struct hash<kmm::IntegerType<Tag, T>>: hash<T> {};
}  // namespace std