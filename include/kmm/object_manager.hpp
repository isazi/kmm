#pragma once

#include <any>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "kmm/types.hpp"

namespace kmm {

using ObjectHandle = std::shared_ptr<std::any>;

class ObjectManager {
  public:
    void create_object(ObjectId, ObjectHandle) const;
    void delete_object(ObjectId) const;
    ObjectHandle get_object(ObjectId) const;

  private:
    mutable std::mutex m_mutex;
    mutable std::unordered_map<ObjectId, ObjectHandle> m_objects;
};

template<typename T>
ObjectHandle make_object(T value) {
    return std::shared_ptr<std::any>(std::move(value));
}

}  // namespace kmm