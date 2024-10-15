#pragma once

#include "kmm/core/reduction.hpp"

namespace kmm {

/**
 *
 */
void execute_reduction(const void* src_buffer, void* dst_buffer, ReductionDef reduction);

}  // namespace kmm