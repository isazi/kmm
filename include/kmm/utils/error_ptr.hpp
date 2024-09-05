#pragma once

#include <exception>
#include <optional>
#include <stdexcept>

namespace kmm {

/**
 * Represents an empty error, thrown when an error is expected but none is present.
 */
class EmptyException final: std::exception {
    const char* what() const noexcept final {
        return "empty `ErrorPtr` encountered where it was expected to hold an exception";
    }
};

/**
 * Class representing an error, encapsulating any arbitrary exception.
 */
class ErrorPtr final {
  public:
    /**
     * Create an empty error.
     */
    ErrorPtr() noexcept = default;

    /**
     * Create an error from the given `std::exception_ptr`.
     */
    explicit ErrorPtr(std::exception_ptr payload) : m_payload(std::move(payload)) {}

    /**
     * Create an error that contains the provided `message`.
     */
    explicit ErrorPtr(const char* message) :
        ErrorPtr(from_exception(std::runtime_error(message))) {}

    /**
     * Create an error that contains the provided `message`.
     */
    explicit ErrorPtr(const std::string& message) : ErrorPtr(message.c_str()) {}

    /**
     * Create an error from the provided exception object.
     */
    template<typename E, typename = std::enable_if_t<std::is_base_of_v<std::exception, E>>>
    explicit ErrorPtr(E&& error) : ErrorPtr(from_exception(std::forward<E>(error))) {}

    /**
     * Create an error from the provided exception object.
     */
    template<typename E>
    static ErrorPtr from_exception(E&& exception) {
        return ErrorPtr {std::make_exception_ptr(std::forward<E>(exception))};
    }

    /**
     * Create an error the current handled error in a catch block.
     */
    static ErrorPtr from_current_exception() {
        return ErrorPtr {std::current_exception()};
    }

    /**
     * Returns `true` if this object contains a valid exception, otherwise return `false`.
     */
    bool has_value() const noexcept {
        return bool(m_payload);
    }

    /**
     * Alias for `has_value`.
     */
    explicit operator bool() const noexcept {
        return has_value();
    }

    /**
     * Throw the exception encapsulated inside this `ErrorPtr`.
     */
    void rethrow() const {
        rethrow_if_present();
        throw EmptyException();
    }

    /**
     * Throw the exception encapsulated inside this `ErrorPtr` if it exists.
     */
    void rethrow_if_present() const {
        if (has_value()) {
            std::rethrow_exception(m_payload);
        }
    }

    /**
     * If the contained exception is of type `E`, apply the provided function `fun` to the
     * exception. Otherwise, return the provided default value.
     */
    template<
        typename E = std::exception,
        typename F,
        typename T = std::invoke_result_t<F, const E&>>
    T map(F fun, T default_value = {}) const {
        try {
            if (m_payload) {
                std::rethrow_exception(m_payload);
            }
        } catch (const E& e) {
            return fun(e);
        } catch (...) {
        }

        return default_value;
    }

    /**
     * Gets a description of the contained exception.
     */
    std::string what() const {
        return map([](const auto& e) { return e.what(); }, "unknown exception");
    }

    /**
     * If the contained exception is of type `E`, return the exception. Otherwise, return `nullopt`.
     */
    template<typename E>
    std::optional<E> get_if() const {
        return this->template map<E>([](const auto& e) { return std::optional<E> {e}; });
    }

    /**
     * Gets the contained exception pointer.
     */
    const std::exception_ptr& get_exception_ptr() const {
        return m_payload;
    }

  private:
    std::exception_ptr m_payload;
};

inline bool operator==(const ErrorPtr& lhs, const ErrorPtr& rhs) {
    return lhs.get_exception_ptr() == rhs.get_exception_ptr();
}

inline bool operator!=(const ErrorPtr& lhs, const ErrorPtr& rhs) {
    return !(lhs == rhs);
}

}  // namespace kmm