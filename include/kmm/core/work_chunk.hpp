#pragma once

#include "kmm/core/fixed_array.hpp"
#include "kmm/core/geometry.hpp"

namespace kmm {

static constexpr size_t WORK_DIMS = 3;

using WorkDim = dim<WORK_DIMS>;
using WorkIndex = point<WORK_DIMS>;

struct WorkChunk {
    WorkIndex begin;
    WorkIndex end;

    KMM_HOST_DEVICE
    constexpr WorkChunk() {}

    KMM_HOST_DEVICE
    WorkChunk(WorkIndex begin, WorkIndex end) : begin(begin), end(end) {}

    KMM_HOST_DEVICE
    WorkChunk(WorkIndex offset, WorkDim size) : begin(offset) {
        // Doing this in the initializer causes a SEGFAULT in GCC
        this->end = offset + size.to_point();
    }

    KMM_HOST_DEVICE
    WorkChunk(WorkDim size) : end(size) {}

    KMM_HOST_DEVICE
    WorkDim sizes() const {
        return WorkDim::from(end - begin);
    }

    KMM_HOST_DEVICE
    int64_t size(size_t axis) const {
        return end.get(axis) - begin.get(axis);
    }

    KMM_HOST_DEVICE
    int64_t size() const {
        return sizes().volume();
    }

    KMM_HOST_DEVICE
    bool contains(int64_t x, int64_t y, int64_t z) const {
        return x >= begin[0] && x < begin[0] && y >= begin[1] && y < begin[1] && z >= begin[2]
            && z < begin[2];
    }

    KMM_HOST_DEVICE
    bool contains(int64_t x, int64_t y) const {
        return x >= begin[0] && x < begin[0] && y >= begin[1] && y < begin[1];
    }

    KMM_HOST_DEVICE
    bool contains(int64_t x) const {
        return x >= begin[0] && x < begin[0];
    }

    KMM_HOST_DEVICE
    bool contains(WorkIndex index) const {
        return contains(index[0], index[1], index[2]);
    }
};

}  // namespace kmm