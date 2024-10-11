#pragma once

#include "kmm/utils/error_ptr.hpp"

namespace kmm {

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

    Result() : m_has_value(true) {}

    Result(error_type error) : m_has_value(false) {
        m_error = std::move(error);
    }

    template<typename U = value_type>
    Result(Result<U>& that) : m_has_value(that.m_has_value) {
        if (!m_has_value) {
            m_error = that.m_storage.error;
        }
    }

    template<typename U = value_type>
    Result(const Result<U>& that) : m_has_value(that.m_has_value) {
        if (!m_has_value) {
            m_error = that.m_storage.error;
        }
    }

    template<typename U = value_type>
    Result(Result<U>&& that) noexcept : m_has_value(that.m_has_value) {
        if (!m_has_value) {
            m_error = std::move(that.m_storage.error);
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
            fun();
        } catch (...) {
            return ErrorPtr::from_current_exception();
        }
    }

    Result& operator=(Result&& that) noexcept {
        if (m_has_value && that.m_has_value) {
        } else {
            if (!m_has_value) {
                m_error = {};
            }

            m_has_value = false;

            if (!that.m_has_value) {
                m_error = std::move(that.m_error);
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
     * Returns the value contained within. Rethrows the exception if the result contains an error.
     */
    void value() const {
        rethrow_if_error();
    }

    /**
     * Returns the value contained within. Rethrows the exception if the result contains an error.
     */
    void operator*() const {
        return value();
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
        if (!has_value()) {
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
    error_type m_error;
    bool m_has_value;
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