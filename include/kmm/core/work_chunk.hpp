#pragma once

#include "kmm/core/fixed_array.hpp"
#include "kmm/core/geometry.hpp"

namespace kmm {

/**
 * Constant for the number of dimensions in the work space.
 */
static constexpr size_t WORK_DIMS = 3;

/**
 * Type alias for the size of the work space.
 */
using WorkDim = dim<WORK_DIMS>;

/**
 * Type alias for the index type used in the work space.
 */
using WorkIndex = point<WORK_DIMS>;

struct WorkChunk {
    WorkIndex begin;  ///< The starting index of the work chunk.
    WorkIndex end;  ///< The ending index of the work chunk.

    /**
     * Initializes an empty chunk.
     */
    KMM_HOST_DEVICE
    constexpr WorkChunk() {}

    /**
     * Constructs a chunk with a given begin and end index.
     */
    KMM_HOST_DEVICE
    WorkChunk(WorkIndex begin, WorkIndex end) : begin(begin), end(end) {}

    /**
     * Constructs a chunk with a given offset and size.
     */
    KMM_HOST_DEVICE
    WorkChunk(WorkIndex offset, WorkDim size) : begin(offset) {
        // Doing this in the initializer causes a SEGFAULT in GCC
        this->end = offset + size.to_point();
    }

    /**
     * Constructs a chunk with a given size and starting at the origin.
     */
    KMM_HOST_DEVICE
    WorkChunk(WorkDim size) : end(size) {}

    /**
     * Gets the sizes of the work chunk in each dimension.
     */
    KMM_HOST_DEVICE
    WorkDim sizes() const {
        return WorkDim::from(end - begin);
    }

    /**
     * Gets the size of the work chunk along a specific axis.
     */
    KMM_HOST_DEVICE
    int64_t size(size_t axis) const {
        return axis < WORK_DIMS ? end[axis] - begin[axis] : 1;
    }

    /**
     * Gets the total size (volume) of the work chunk.
     */
    KMM_HOST_DEVICE
    int64_t size() const {
        return sizes().volume();
    }

    /**
     * Checks if a given 3D point is contained within the work chunk.
     */
    KMM_HOST_DEVICE
    bool contains(int64_t x, int64_t y, int64_t z) const {
        return (x >= begin[0] && x < end[0]) &&  //
            (y >= begin[1] && y < end[1]) &&  //
            (z >= begin[2] && z < end[2]);
    }

    /**
     * Checks if a given 2D point is contained within the work chunk.
     */
    KMM_HOST_DEVICE
    bool contains(int64_t x, int64_t y) const {
        return x >= begin[0] && x < end[0] && y >= begin[1] && y < end[1];
    }

    /**
     * Checks if a given 1D point is contained within the work chunk.
     */
    KMM_HOST_DEVICE
    bool contains(int64_t x) const {
        return x >= begin[0] && x < end[0];
    }

    /**
     * Checks if a multidimensional point is contained within the work chunk.
     */
    template<size_t N>
    KMM_HOST_DEVICE bool contains(point<N> p) const {
        bool result = true;

        for (size_t i = 0; i < N && i < WORK_DIMS; i++) {
            result &= p[i] >= begin[i] && p[i] < end[i];
        }

        return result;
    }
};

}  // namespace kmm