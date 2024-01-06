#include <cstdlib>

#include "kmm/host/memory.hpp"

namespace kmm {

class HostAllocationImpl: public HostAllocation {
  public:
    explicit HostAllocationImpl(size_t nbytes);
    ~HostAllocationImpl();
    void* data() const final;
    size_t size() const final;

  private:
    void* m_data;
    size_t m_nbytes;
};

HostAllocationImpl::HostAllocationImpl(size_t nbytes) : m_nbytes(nbytes) {
    size_t alignment = alignof(std::max_align_t);

    if (nbytes > 0) {
        if (nbytes % alignment != 0) {
            nbytes += alignment - nbytes % alignment;
        }

        m_data = ::aligned_alloc(alignment, nbytes);
        KMM_ASSERT(m_data != nullptr);
    }
}

HostAllocationImpl::~HostAllocationImpl() {
    free(m_data);
}

void* HostAllocationImpl::data() const {
    return m_data;
}

size_t HostAllocationImpl::size() const {
    return m_nbytes;
}

HostMemory::HostMemory(std::shared_ptr<ThreadPool> pool, size_t max_bytes) :
    m_pool(std::move(pool)),
    m_bytes_remaining(max_bytes) {}

std::optional<std::unique_ptr<MemoryAllocation>> HostMemory::allocate(
    MemoryId id,
    size_t num_bytes) {
    KMM_ASSERT(id == 0);
    if (m_bytes_remaining >= num_bytes) {
        m_bytes_remaining -= num_bytes;
        return std::make_unique<HostAllocationImpl>(num_bytes);
    } else {
        return std::nullopt;
    }
}

void HostMemory::deallocate(MemoryId id, std::unique_ptr<MemoryAllocation> allocation) {
    KMM_ASSERT(id == 0);
    auto& alloc = dynamic_cast<HostAllocation&>(*allocation);
    m_bytes_remaining += alloc.size();
}

bool HostMemory::is_copy_possible(MemoryId src_id, MemoryId dst_id) {
    return src_id == 0 && dst_id == 0;
}

void HostMemory::copy_async(
    MemoryId src_id,
    const MemoryAllocation* src_alloc,
    size_t src_offset,
    MemoryId dst_id,
    const MemoryAllocation* dst_alloc,
    size_t dst_offset,
    size_t num_bytes,
    Completion completion) {
    const auto& src_host = dynamic_cast<const HostAllocation&>(*src_alloc);
    const auto& dst_host = dynamic_cast<const HostAllocation&>(*dst_alloc);

    KMM_ASSERT(src_id == 0);
    KMM_ASSERT(dst_id == 0);
    KMM_ASSERT(num_bytes <= src_host.size());
    KMM_ASSERT(num_bytes <= dst_host.size());
    KMM_ASSERT(src_offset <= src_host.size() - num_bytes);
    KMM_ASSERT(dst_offset <= dst_host.size() - num_bytes);

    m_pool->submit_copy(
        static_cast<const char*>(src_host.data()) + src_offset,
        static_cast<char*>(dst_host.data()) + dst_offset,
        num_bytes,
        std::move(completion));
}

void HostMemory::fill_async(
    MemoryId dst_id,
    const MemoryAllocation* dst_alloc,
    size_t dst_offset,
    size_t num_bytes,
    std::vector<uint8_t> fill_pattern,
    Completion completion) {
    const auto& dst_host = dynamic_cast<const HostAllocation&>(*dst_alloc);

    KMM_ASSERT(dst_id == 0);
    KMM_ASSERT(num_bytes <= dst_host.size());
    KMM_ASSERT(dst_offset <= dst_host.size() - num_bytes);

    m_pool->submit_fill(
        static_cast<char*>(dst_host.data()) + dst_offset,
        num_bytes,
        std::move(fill_pattern),
        std::move(completion));
}

}  // namespace kmm