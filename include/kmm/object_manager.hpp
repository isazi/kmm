#pragma once

#include <any>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <variant>

#include "kmm/executor.hpp"
#include "kmm/object.hpp"
#include "kmm/types.hpp"

namespace kmm {

class ObjectManager {
  public:
    void create_object(ObjectId, ObjectHandle) const;
    void poison_object(ObjectId, TaskError reason) const;
    ObjectHandle get_object(ObjectId) const;
    void delete_object(ObjectId) const;

  private:
    mutable std::mutex m_mutex;
    mutable std::unordered_map<ObjectId, TaskResult> m_objects;
};

}  // namespace kmm