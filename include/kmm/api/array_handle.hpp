#pragma once

#include "kmm/core/geometry.hpp"
#include "kmm/core/identifiers.hpp"

namespace kmm {

class Worker;

template<size_t N>
class ArrayHandle: public std::enable_shared_from_this<ArrayHandle<N>> {
    KMM_NOT_COPYABLE_OR_MOVABLE(ArrayHandle)

  public:
    ArrayHandle(Worker& worker, std::pair<DataDistribution<N>, std::vector<BufferId>> distribution);
    ~ArrayHandle();

    BufferId buffer(size_t index) const;
    void copy_bytes(void* dest_addr, size_t element_size) const;
    void synchronize() const;

    const DataDistribution<N>& distribution() const {
        return m_distribution;
    }

    const Worker& worker() const {
        return *m_worker;
    }

  private:
    DataDistribution<N> m_distribution;
    std::shared_ptr<Worker> m_worker;
    std::vector<BufferId> m_buffers;
};

}  // namespace kmm