#pragma once

#include "kmm/block.hpp"
#include "kmm/executor.hpp"
#include "kmm/types.hpp"

namespace kmm {

class BlockManager {
  public:
    void insert_block(
        BlockId,
        std::shared_ptr<BlockHeader> header,
        std::optional<BufferId> buffer_id);
    void poison_block(BlockId, TaskError);
    std::optional<BufferId> delete_block(BlockId);

    std::shared_ptr<BlockHeader> get_block_header(BlockId) const;
    std::optional<BufferId> get_block_buffer(BlockId) const;

  private:
    struct Entry {
        std::shared_ptr<BlockHeader> header;
        std::optional<BufferId> buffer_id;
    };

    void insert_entry(BlockId, std::variant<Entry, TaskError>);

    std::unordered_map<BlockId, std::variant<Entry, TaskError>> m_entries;
};

}  // namespace kmm