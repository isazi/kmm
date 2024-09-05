#pragma once

#include <vector>

#include "buffer.hpp"

namespace kmm {

struct TaskContext {
    std::vector<BufferAccessor> accessors;
};

/**
 * Exception throw if
 */
class InvalidExecutionContext: public std::exception {
  public:
    InvalidExecutionContext(const std::type_info& expected, const std::type_info& gotten);
    const char* what() const noexcept override;

  private:
    std::string m_message;
};

struct ExecutionContext {
    virtual ~ExecutionContext() = default;

    template<typename T>
    T* cast_if() {
        return dynamic_cast<T*>(this);
    }

    template<typename T>
    const T* cast_if() const {
        return dynamic_cast<const T*>(this);
    }

    template<typename T>
    T& cast() {
        if (auto* ptr = this->template cast_if<T>()) {
            return *ptr;
        }

        throw InvalidExecutionContext(typeid(T), typeid(*this));
    }

    template<typename T>
    const T& cast() const {
        if (auto* ptr = this->template cast_if<T>()) {
            return *ptr;
        }

        throw InvalidExecutionContext(typeid(T), typeid(*this));
    }

    template<typename T>
    bool is() const {
        return this->template cast_if<T>() != nullptr;
    }
};

struct HostContext: public ExecutionContext {};

class Task {
  public:
    virtual ~Task() = default;
    virtual void execute(ExecutionContext& device, TaskContext context) = 0;
};

}  // namespace kmm