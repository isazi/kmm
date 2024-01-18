#pragma once

#include <exception>
#include <optional>
#include <stdexcept>
#include <utility>

#include "kmm/panic.hpp"

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
     * Returns `true` if the contained exception is of type `E, otherwise returns `false`.
     */
    template<typename E>
    bool is() const {
        return this->template map<E>([](const auto& e) { return true; });
    }

    const std::type_info& type() const {
        return *map([](const auto& e) { return &typeid(e); }, &typeid(nullptr));
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

template<typename T = void>
class Result final {
  public:
    using value_type = T;
    using error_type = ErrorPtr;

    template<typename... Args>
    Result(Args&&... args) : m_has_value(true) {
        m_storage.initialize_value(std::forward<Args>(args)...);
    }

    Result(value_type value) : m_has_value(true) {
        m_storage.initialize_value(std::move(value));
    }

    Result(error_type error) : m_has_value(false) {
        m_storage.initialize_error(std::move(error));
    }

    template<typename U = value_type>
    Result(Result<U>& that) : m_has_value(that.m_has_value) {
        if (m_has_value) {
            m_storage.initialize_value(that.m_storage.value);
        } else {
            m_storage.initialize_error(that.m_storage.error);
        }
    }

    template<typename U = value_type>
    Result(const Result<U>& that) : m_has_value(that.m_has_value) {
        if (m_has_value) {
            m_storage.initialize_value(that.m_storage.value);
        } else {
            m_storage.initialize_error(that.m_storage.error);
        }
    }

    template<typename U = value_type>
    Result(Result<U>&& that) noexcept : m_has_value(that.m_has_value) {
        if (m_has_value) {
            m_storage.initialize_value(std::move(that.m_storage.value));
        } else {
            m_storage.initialize_error(std::move(that.m_storage.error));
        }
    }

    ~Result() {
        if (m_has_value) {
            m_storage.destroy_value();
        } else {
            m_storage.destroy_error();
        }
    }

    template<typename E>
    static Result from_error(E&& error) noexcept {
        return ErrorPtr(std::forward<E>(error));
    }

    static Result from_empty() noexcept {
        return ErrorPtr();
    }

    static Result from_current_exception() noexcept {
        return ErrorPtr::from_current_exception();
    }

    template<typename F>
    static Result try_catch(F fun) noexcept {
        try {
            return T {fun()};
        } catch (...) {
            return ErrorPtr::from_current_exception();
        }
    }

    Result& operator=(Result&& that) noexcept {
        static_assert(std::is_nothrow_move_constructible_v<value_type>, "T must be movable");

        if (m_has_value && that.m_has_value) {
            m_storage.assign_value(std::move(that.m_storage.value));
        } else {
            if (m_has_value) {
                m_storage.destroy_value();
            } else {
                m_storage.destroy_error();
            }

            m_has_value = false;

            if (that.m_has_value) {
                m_storage.initialize_value(std::move(that.m_storage.value));
            } else {
                m_storage.initialize_error(std::move(that.m_storage.error));
            }

            m_has_value = that.m_has_value;
        }

        return *this;
    }

    template<typename U>
    Result& operator=(Result<U>&& that) noexcept {
        *this = Result(std::move(that));
        return *this;
    }

    template<typename U>
    Result& operator=(const Result<U>& that) {
        *this = Result(that);
        return *this;
    }

    template<typename U>
    Result& operator=(Result<U>& that) {
        *this = Result(that);
        return *this;
    }

    /**
     * Returns `true` if this result contains value (not an error) and `false` otherwise.
     */
    bool has_value() const noexcept {
        return m_has_value;
    }

    /**
     * Alias for `has_value()`
     */
    explicit operator bool() const {
        return has_value();
    }

    /**
     * Return a pointer the value contained within, or `nullptr` if the result contains an error.
     */
    value_type* value_if_present() {
        return has_value() ? &m_storage.value : nullptr;
    }

    /**
     * Return a pointer the value contained within, or `nullptr` if the result contains an error.
     */
    const value_type* value_if_present() const {
        return has_value() ? &m_storage.value : nullptr;
    }

    /**
     * Returns the value contained within. Rethrows the exception if the result contains an error.
     */
    const value_type& value() const& {
        rethrow_if_error();
        return m_storage.value;
    }

    /**
     * Returns the value contained within. Rethrows the exception if the result contains an error.
     */
    value_type& value() & {
        rethrow_if_error();
        return m_storage.value;
    }

    /**
     * Returns the value contained within. Rethrows the exception if the result contains an error.
     */
    value_type&& value() && {
        rethrow_if_error();
        return std::move(m_storage.value);
    }

    /**
     * Returns the value contained within. Rethrows the exception if the result contains an error.
     */
    const value_type& operator*() const& {
        return value();
    }

    /**
     * Returns the value contained within. Rethrows the exception if the result contains an error.
     */
    value_type& operator*() & {
        return value();
    }

    /**
     * Returns the value contained within. Rethrows the exception if the result contains an error.
     */
    value_type&& operator*() && {
        return std::move(*this).value();
    }

    /**
     * Returns `true` if this result contains an error and `false` otherwise.
     */
    bool has_error() const {
        return !has_value();
    }

    /**
     * Throw the exception encapsulated inside this result. Throws `EmptyException` if this result
     * does not contain an error.
     */
    void rethrow_error() const {
        if (has_error()) {
            m_storage.error.rethrow_if_present();
        }

        throw EmptyException();
    }

    /**
     * Throw the exception encapsulated inside this result only if this result contains an error.
     */
    void rethrow_if_error() const {
        if (!has_value()) {
            m_storage.error.rethrow_if_present();
            throw EmptyException();
        }
    }

    /**
     * Returns the error contained within. Throws `EmptyException` if the result contains no error.
     */
    const error_type& error() const {
        if (!has_error()) {
            throw EmptyException();
        }

        return m_storage.error;
    }

    /**
     * Returns a pointer to the error contained within, or `nullptr` if this result has no error.
     */
    const error_type* error_if_present() const {
        return has_error() ? &m_storage.error : nullptr;
    }

    template<typename R, typename U>
    friend bool operator==(const Result<R>& lhs, const Result<U>& rhs) {
        if (lhs.m_has_value && rhs.m_has_value) {
            return lhs.m_storage.value == rhs.m_storage.value;
        } else if (lhs.has_error() && rhs.has_error()) {
            return lhs.m_storage.error == rhs.m_storage.error;
        } else {
            return false;
        }
    }

  private:
    union storage_type {
        storage_type() {}
        ~storage_type() {}

        template<typename E>
        void initialize_error(E&& err) {
            new (&this->error) ErrorPtr(std::forward<E>(err));
        }

        template<typename... Args>
        void initialize_value(Args&&... args) {
            new (&this->value) T(std::forward<Args>(args)...);
        }

        void assign_value(T&& new_value) {
            this->value = std::move(new_value);
        }

        void destroy_value() {
            this->value.~T();
        }

        void destroy_error() {
            this->error.~ErrorPtr();
        }

        value_type value;
        error_type error;
    };

    storage_type m_storage;
    bool m_has_value;
};

template<>
class Result<void> final {
  public:
    using value_type = void;
    using error_type = ErrorPtr;

    Result() = default;

    Result(error_type error) : m_error(std::move(error)) {}

    Result(Result&) = default;
    Result(const Result&) = default;
    Result(Result&&) noexcept = default;

    template<typename U = value_type>
    Result(Result<U>& that) {
        if (that.has_error()) {
            m_error = *that.error_if_present();
        }
    }

    template<typename U = value_type>
    Result(const Result<U>& that) {
        if (that.has_error()) {
            m_error = *that.error_if_present();
        }
    }

    template<typename U = value_type>
    Result(Result<U>&& that) noexcept {
        if (that.has_error()) {
            m_error = *that.error_if_present();
        }
    }

    template<typename E>
    static Result from_error(E&& error) noexcept {
        return ErrorPtr(std::forward<E>(error));
    }

    static Result from_empty() noexcept {
        return ErrorPtr(EmptyException());
    }

    static Result from_current_exception() noexcept {
        return ErrorPtr::from_current_exception();
    }

    template<typename F>
    static Result try_catch(F fun) noexcept {
        try {
            fun();
            return Result {};
        } catch (...) {
            return ErrorPtr::from_current_exception();
        }
    }

    Result& operator=(Result& that) = default;
    Result& operator=(const Result& that) = default;
    Result& operator=(Result&& that) noexcept = default;

    template<typename U>
    Result& operator=(Result<U>&& that) noexcept {
        *this = Result(std::move(that));
        return *this;
    }

    template<typename U>
    Result& operator=(const Result<U>& that) {
        *this = Result(that);
        return *this;
    }

    template<typename U>
    Result& operator=(Result<U>& that) {
        *this = Result(that);
        return *this;
    }

    /**
     * Returns `true` if this result contains value (not an error) and `false` otherwise.
     */
    bool has_value() const noexcept {
        return !m_error.has_value();
    }

    /**
     * Alias for `has_value()`
     */
    explicit operator bool() const {
        return has_value();
    }

    /**
     * Rethrows the exception if the result contains an error.
     */
    void value() const {
        rethrow_if_error();
    }

    /**
     * Returns `true` if this result contains an error and `false` otherwise.
     */
    bool has_error() const {
        return !has_value();
    }

    /**
     * Throw the exception encapsulated inside this result. Throws `EmptyException` if this result
     * does not contain an error.
     */
    void rethrow_error() const {
        if (has_error()) {
            m_error.rethrow_if_present();
        }

        throw EmptyException();
    }

    /**
     * Throw the exception encapsulated inside this result only if this result contains an error.
     */
    void rethrow_if_error() const {
        if (has_error()) {
            m_error.rethrow_if_present();
            throw EmptyException();
        }
    }

    /**
     * Returns the error contained within. Throws `EmptyException` if the result contains no error.
     */
    const error_type& error() const {
        if (!has_error()) {
            throw EmptyException();
        }

        return m_error;
    }

    /**
     * Returns a pointer to the error contained within, or `nullptr` if this result has no error.
     */
    const error_type* error_if_present() const {
        return has_error() ? &m_error : nullptr;
    }

    friend bool operator==(const Result<void>& lhs, const Result<void>& rhs) {
        return lhs.m_error == rhs.m_error;
    }

  private:
    error_type m_error;
};

template<typename T, typename U>
bool operator!=(const Result<T>& lhs, const Result<U>& rhs) {
    return !operator==(lhs, rhs);
}

template<typename F, typename T = std::invoke_result_t<F>>
Result<T> try_catch(F&& fun) {
    return Result<T>::try_catch(std::forward<F>(fun));
}

}  // namespace kmm