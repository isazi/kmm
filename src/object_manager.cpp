#include "kmm/object_manager.hpp"

namespace kmm {

void ObjectManager::create_object(ObjectId id, ObjectHandle object) const {
    std::lock_guard guard {m_mutex};
    m_objects.insert({id, std::move(object)});
}

void ObjectManager::delete_object(ObjectId id) const {
    std::lock_guard guard {m_mutex};

    if (auto it = m_objects.find(id); it != m_objects.end()) {
        m_objects.erase(id);
    }
}

ObjectHandle ObjectManager::get_object(ObjectId id) const {
    std::lock_guard guard {m_mutex};

    if (auto it = m_objects.find(id); it != m_objects.end()) {
        return it->second;
    }

    throw std::runtime_error("invalid object id");
}

}  // namespace kmm