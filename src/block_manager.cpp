#include "kmm/block_manager.hpp"

namespace kmm {

void BlockManager::insert_block(
    BlockId id,
    std::shared_ptr<BlockHeader> header,
    std::optional<BufferId> buffer_id) {
    insert_entry(
        id,
        Entry {
            .header = std::move(header),
            .buffer_id = buffer_id,
        });
}

void BlockManager::poison_block(BlockId id, TaskError error) {
    insert_entry(id, std::move(error));
}

void BlockManager::insert_entry(BlockId id, std::variant<Entry, TaskError> entry) {
    auto [_, success] = m_entries.insert({id, std::move(entry)});

    if (!success) {
        throw std::runtime_error(
            fmt::format("cannot insert block {}, block with same identifier already exists", id));
    }
}

std::optional<BufferId> BlockManager::delete_block(BlockId id) {
    auto it = m_entries.find(id);

    if (it == m_entries.end()) {
        return std::nullopt;
    }

    std::optional<BufferId> result = std::nullopt;
    if (auto* entry = std::get_if<Entry>(&it->second)) {
        result = entry->buffer_id;
    }

    m_entries.erase(it);
    return result;
}

std::shared_ptr<BlockHeader> BlockManager::get_block_header(BlockId id) const {
    auto it = m_entries.find(id);

    if (it != m_entries.end()) {
        if (const auto* entry = std::get_if<Entry>(&it->second)) {
            return entry->header;
        } else if (const auto* error = std::get_if<TaskError>(&it->second)) {
            throw std::runtime_error(error->get());
        }
    }

    throw std::runtime_error(fmt::format("unknown block: {}", id));
}

std::optional<BufferId> BlockManager::get_block_buffer(BlockId id) const {
    auto it = m_entries.find(id);

    if (it != m_entries.end()) {
        if (const auto* entry = std::get_if<Entry>(&it->second)) {
            return entry->buffer_id;
        }
    }

    return std::nullopt;
}
}  // namespace kmm