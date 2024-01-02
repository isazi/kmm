#pragma once

#include <memory>
#include <optional>
#include <unordered_map>

#include "kmm/block.hpp"
#include "kmm/identifiers.hpp"
#include "kmm/result.hpp"

namespace kmm {

struct BlockMetadata {
    std::shared_ptr<BlockHeader> header;
    std::optional<BufferId> buffer_id;
    MemoryId home_memory;
};

class BlockManager {
  public:
    /**
     * Insert a new block with a given `id` and `header`. If this block has an associated block,
     * then `buffer_id` should be its identifier.
     */
    void insert_block(
        BlockId id,
        std::shared_ptr<BlockHeader> header,
        MemoryId home_memory,
        std::optional<BufferId> buffer_id);

    /**
     * Mark a block as poisoned.
     */
    void poison_block(BlockId id, ErrorPtr error);

    /**
     * Delete the given block. Returns the buffer associated with that block since the buffer
     * still needs to be deleted by somebody afterwards.
     */
    std::optional<BufferId> delete_block(BlockId);

    /**
     * Returns the metadata for the given block. Throws an exception if the block does not exist.
     */
    const BlockMetadata& get_block(BlockId) const;

    /**
     * Get the buffer associated with the given block. The result might be nullopt if the
     * block does not have a buffer, if the block does not exist, or if the block is poisoned.
     */
    std::optional<BufferId> get_block_buffer(BlockId) const;

  private:
    void insert_entry(BlockId, Result<BlockMetadata>);

    std::unordered_map<BlockId, Result<BlockMetadata>> m_entries;
};

}  // namespace kmm