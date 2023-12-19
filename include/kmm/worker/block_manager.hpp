#pragma once

#include "kmm/block.hpp"
#include "kmm/executor.hpp"
#include "kmm/types.hpp"

namespace kmm {

struct BlockMetadata {
    std::shared_ptr<BlockHeader> header;
    std::optional<BufferId> buffer_id;
    DeviceId home_memory;
};

class BlockManager {
  public:
    void insert_block(
        BlockId,
        std::shared_ptr<BlockHeader> header,
        DeviceId home_memory,
        std::optional<BufferId> buffer_id);

    void poison_block(BlockId, TaskError);
    std::optional<BufferId> delete_block(BlockId);
    const BlockMetadata& get_block(BlockId) const;
    std::optional<BufferId> get_block_buffer(BlockId) const;

  private:
    void insert_entry(BlockId, std::variant<BlockMetadata, TaskError>);

    std::unordered_map<BlockId, std::variant<BlockMetadata, TaskError>> m_entries;
};

}  // namespace kmm