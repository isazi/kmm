#pragma once

#include "kmm/core/fixed_array.hpp"
#include "kmm/core/geometry.hpp"

namespace kmm {

/**
 * Constant for the number of dimensions in the work space.
 */
static constexpr size_t ND_DIMS = 3;

/**
 * Type alias for the index type used in the work space.
 */
using NDIndex = Point<ND_DIMS>;

/**
 * Type alias for the size of the work space.
 */
using NDDim = Dim<ND_DIMS>;

struct NDRange {
    NDIndex begin;  ///< The starting index of the work chunk.
    NDIndex end;  ///< The ending index of the work chunk.

    /**
     * Initializes an empty chunk.
     */
    KMM_HOST_DEVICE
    NDRange(int64_t x = 1, int64_t y = 1, int64_t z = 1) : begin(0, 0, 0), end(x, y, z) {}

    /**
     * Constructs a chunk with a given begin and end index.
     */
    KMM_HOST_DEVICE
    explicit NDRange(NDIndex begin, NDIndex end) : begin(begin), end(end) {}

    /**
     * Constructs a chunk with a given offset and size.
     */
    KMM_HOST_DEVICE
    explicit NDRange(NDIndex offset, NDDim size) : begin(offset) {
        // Doing this in the initializer causes a SEGFAULT in GCC
        this->end = offset + size.to_point();
    }

    /**
     * Constructs a chunk with a given size and starting at the origin.
     */
    KMM_HOST_DEVICE
    NDRange(NDDim size) : end(size) {}

    /**
     * Gets the sizes of the work chunk in each dimension.
     */
    KMM_HOST_DEVICE
    NDDim sizes() const {
        return NDDim::from(end - begin);
    }

    /**
     * Gets the size of the work chunk along a specific axis.
     */
    KMM_HOST_DEVICE
    int64_t size(size_t axis) const {
        return axis < ND_DIMS ? end[axis] - begin[axis] : 1;
    }

    /**
     * Gets the total size (volume) of the work chunk.
     */
    KMM_HOST_DEVICE
    int64_t size() const {
        return sizes().volume();
    }

    /**
     * Gets the total size (volume) of the work chunk.
     */
    KMM_HOST_DEVICE
    bool is_empty() const {
        return sizes().is_empty();
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
    KMM_HOST_DEVICE bool contains(Point<N> p) const {
        bool result = true;

        for (size_t i = 0; i < N && i < ND_DIMS; i++) {
            result &= p[i] >= begin[i] && p[i] < end[i];
        }

        return result;
    }
};

}  // namespace kmm