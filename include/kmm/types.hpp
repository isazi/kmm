#pragma once

#include <string>

namespace kmm {

template<typename T, typename Tag = void>
class Identifier {
  public:
    explicit constexpr Identifier(T value) : m_value(value) {}

    static constexpr Identifier invalid() {
        return Identifier(std::numeric_limits<T>::max());
    }

    T get() const {
        return m_value;
    }

    operator T() const {
        return get();
    }

    bool operator==(const Identifier& that) const {
        return m_value == that.m_value;
    }

    bool operator<(const Identifier& that) const {
        return m_value < that.m_value;
    }

    bool operator!=(const Identifier& that) const {
        return !(*this == that);
    }

    bool operator<=(const Identifier& that) const {
        return !(this > that);
    }

    bool operator>(const Identifier& that) const {
        return that < *this;
    }

    bool operator>=(const Identifier& that) const {
        return that <= *this;
    }

    Identifier operator++() {
        return Identifier(++m_value);
    }

    Identifier operator++(int) {
        return Identifier(m_value++);
    }

  private:
    T m_value;
};

using index_t = int;
using DeviceId = Identifier<uint8_t, struct DeviceTag>;
using OperationId = Identifier<uint64_t, struct OperationTag>;
using BufferId = Identifier<uint64_t, struct BufferTag>;
using PhysicalBufferId = Identifier<uint64_t, struct PhysicalBufferTag>;
using ObjectId = Identifier<uint64_t, struct ObjectTag>;

struct BufferLayout {
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
    BufferId buffer_id;
    AccessMode mode;
};

class RuntimeImpl;
class Runtime;
class Object;

}  // namespace kmm

namespace std {
template<typename T, typename Tag>
struct hash<kmm::Identifier<T, Tag>>: hash<T> {};
}  // namespace std