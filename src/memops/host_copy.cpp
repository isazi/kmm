#include <cstdint>
#include <cstring>
#include <functional>

#include "kmm/core/reduction.hpp"
#include "kmm/memops/host_copy.hpp"

namespace kmm {

template<typename T, typename F>
void execute_reduction(
    const T* src_buffer,
    T* dst_buffer,
    F combine,
    size_t num_partials,
    size_t num_results) {
    for (size_t i = 0; i < num_results; i++) {
        dst_buffer[i] = src_buffer[i];

        for (size_t j = 1; j < num_partials; j++) {
            dst_buffer[i] = combine(dst_buffer[i], src_buffer[num_results * j + i]);
        }
    }
}

void execute_reduction(
    const void* src_buffer,
    void* dst_buffer,
    Reduction reduction,
    size_t num_partials,
    size_t num_results) {
    switch (reduction.data_type) {
        case DataType::Int8:
            switch (reduction.op) {
                case ReductionOp::Sum:
                    return execute_reduction(
                        (const int8_t*)src_buffer,
                        (int8_t*)dst_buffer,
                        std::plus<int8_t>(),
                        num_partials,
                        num_results);
                case ReductionOp::Product:
                    break;
                case ReductionOp::Min:
                    break;
                case ReductionOp::Max:
                    break;
                case ReductionOp::BitAnd:
                    break;
                case ReductionOp::BitOr:
                    break;
            }
        case DataType::Int16:
            break;
        case DataType::Int32:
            break;
        case DataType::Int64:
            break;
        case DataType::Uint8:
            break;
        case DataType::Uint16:
            break;
        case DataType::Uint32:
            break;
        case DataType::Uint64:
            break;
        case DataType::Float16:
            break;
        case DataType::Float32:
            break;
        case DataType::Float64:
            break;
        case DataType::Complex16:
            break;
        case DataType::Complex32:
            break;
        case DataType::Complex64:
            break;
    }
}

void execute_copy(const void* src_buffer, void* dst_buffer, CopyDescription copy_description) {}

}  // namespace kmm