#pragma once

#include <memory>
#include <stdexcept>
#include <string>

namespace kmm {

template<typename T>
class ObjectImpl;

class Object {
  public:
    virtual ~Object() = default;
    virtual const std::type_info& type_info() const = 0;
    virtual std::string type_name() const = 0;

    template<typename T>
    const T* get_if() const {
        if (auto ptr = dynamic_cast<const ObjectImpl<T>*>(this)) {
            return &ptr->value();
        } else {
            return nullptr;
        }
    }

    template<typename T>
    const T& get() const {
        if (auto ptr = get_if<T>()) {
            return *ptr;
        } else {
            throw std::runtime_error("invalid type");
        }
    }

    template<typename T>
    bool is() const {
        return get_if<T>() != nullptr;
    }
};

template<typename T>
class ObjectImpl: public Object {
  public:
    static_assert(std::is_same<T, std::decay_t<T>>(), "type cannot be reference");

    ObjectImpl() = delete;
    ObjectImpl(const ObjectImpl&) = delete;
    ObjectImpl(ObjectImpl&&) = delete;
    ObjectImpl(T value) : m_value(std::move(value)) {}

    const std::type_info& type_info() const override {
        return typeid(T);
    }

    std::string type_name() const override {
        return typeid(T).name();
    }

    const T& value() const {
        return m_value;
    }

  private:
    T m_value;
};

using ObjectHandle = std::shared_ptr<const Object>;

template<typename T>
ObjectHandle make_object(T&& value) {
    return std::make_shared<ObjectImpl<std::decay_t<T>>>(std::forward<T>(value));
}

template<typename T>
std::shared_ptr<const T> object_cast(ObjectHandle obj) {
    const T* ptr = &obj->get<T>();
    return std::shared_ptr<const T>(std::move(obj), ptr);
}

}  // namespace kmm