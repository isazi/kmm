#pragma once

#include <memory>
#include <type_traits>
#include <utility>

namespace kmm {

/**
 * Interface to notify when an event has occurred
 */
class Notify {
  public:
    virtual ~Notify() noexcept = default;

    /**
     * Called when an event is triggered.
     */
    virtual void notify() const noexcept = 0;
};

/**
 * Implementation of `Notify` that forwards the notify call to a function `F`.
 */
template<typename F>
class NotifyImpl: public Notify {
  public:
    NotifyImpl(F fun = {}) : m_callback(std::move(fun)) {}

    void notify() const noexcept final {
        m_callback();
    }

  private:
    F m_callback;
};

/**
 * Wrapper around a `shared_ptr<Notify>` handle.
 */
class NotifyHandle {
  public:
    NotifyHandle() = default;
    NotifyHandle(std::shared_ptr<Notify> m);
    NotifyHandle(std::unique_ptr<Notify> m);

    template<typename T>
    NotifyHandle(std::shared_ptr<T> m) : NotifyHandle(std::shared_ptr<Notify>(m)) {}

    template<typename T>
    NotifyHandle(std::unique_ptr<T> m) : NotifyHandle(std::shared_ptr<Notify>(std::move(m))) {}

    template<typename F, typename std::enable_if<std::is_invocable<F>::value, int>::type = 0>
    NotifyHandle(F&& callback) :
        NotifyHandle(
            std::make_shared<NotifyImpl<typename std::decay<F>>::type>(std::forward<F>(callback))) {
    }

    ~NotifyHandle();

    /**
     * If the underlying `Notify` object exists, this calls its `notify()` method.
     */
    void notify() const noexcept;

    /**
     * Resets the managed `shared_ptr<Notify>` to null, effectively clearing the
     * notification handler.
     */
    void clear() noexcept;

    /**
     * This calls the `notify()` method on the underlying `Notify` object (if it exists),
     * and then resets the managed `shared_ptr<Notify>` to null.
     */
    void notify_and_clear() noexcept;

  private:
    std::shared_ptr<Notify> m_impl;
};

}  // namespace kmm