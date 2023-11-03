#include "kmm/object_manager.hpp"
#include "kmm/utils.hpp"

namespace kmm {

void ObjectManager::create_object(ObjectId id, ObjectHandle object) const {
    KMM_ASSERT(object);

    std::lock_guard guard {m_mutex};
    m_objects.erase(id);
    m_objects.insert({id, std::move(object)});
}

void ObjectManager::poison_object(ObjectId id, TaskError reason) const {
    std::lock_guard guard {m_mutex};
    m_objects.erase(id);
    m_objects.insert({id, std::move(reason)});
}

void ObjectManager::delete_object(ObjectId id) const {
    std::lock_guard guard {m_mutex};

    if (auto it = m_objects.find(id); it != m_objects.end()) {
        m_objects.erase(id);
    }
}

ObjectHandle ObjectManager::get_object(ObjectId id) const {
    std::lock_guard guard {m_mutex};

    auto it = m_objects.find(id);
    if (it == m_objects.end()) {
        throw std::runtime_error("invalid object id: " + std::to_string(id));
    }

    const auto& result = it->second;
    if (const auto* error = std::get_if<TaskError>(&result)) {
        throw std::runtime_error("object is poisoned: " + error->get());
    }

    return std::get<ObjectHandle>(result);
}

}  // namespace kmm