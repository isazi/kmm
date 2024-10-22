#include "kmm/internals/allocator/base.hpp"

namespace kmm {

struct DirectMemoryAllocator::DeferredDealloc {
    void* addr;
    size_t nbytes;
    DeviceEventSet dependencies;
};

DirectMemoryAllocator::DirectMemoryAllocator(
    std::shared_ptr<CudaStreamManager> streams,
    size_t max_bytes
) :
    m_streams(streams),
    m_bytes_limit(max_bytes),
    m_bytes_in_use(0) {}

DirectMemoryAllocator::~DirectMemoryAllocator() {}

bool DirectMemoryAllocator::allocate(size_t nbytes, void*& addr_out, DeviceEventSet& deps_out) {
    KMM_ASSERT(nbytes > 0);
    make_progress();

    while (true) {
        if (m_bytes_limit - m_bytes_in_use >= nbytes) {
            if (allocate_impl(nbytes, addr_out)) {
                m_bytes_in_use += nbytes;
                return true;
            }
        }

        if (m_pending_deallocs.empty()) {
            return false;
        }

        auto d = m_pending_deallocs.front();
        m_streams->wait_until_ready(d.dependencies);
        m_pending_deallocs.pop_front();
        m_bytes_in_use -= d.nbytes;

        deallocate_impl(d.addr, d.nbytes);
    }
}

void DirectMemoryAllocator::deallocate(void* addr, size_t nbytes, DeviceEventSet deps) {
    make_progress();

    if (m_streams->is_ready(deps)) {
        m_bytes_in_use -= nbytes;
        deallocate_impl(addr, nbytes);
    } else {
        m_pending_deallocs.push_back({addr, nbytes, std::move(deps)});
    }
}

void DirectMemoryAllocator::make_progress() {
    while (!m_pending_deallocs.empty()) {
        auto d = m_pending_deallocs.front();

        if (!m_streams->is_ready(d.dependencies)) {
            break;
        }

        m_pending_deallocs.pop_front();

        m_bytes_in_use -= d.nbytes;
        deallocate_impl(d.addr, d.nbytes);
    }
}

}  // namespace kmm