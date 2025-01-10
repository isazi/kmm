#pragma once

#include "kmm/memops/types.hpp"

namespace kmm {

/**
 *
 */
void execute_reduction(const void* src_buffer, void* dst_buffer, ReductionDef reduction);

}  // namespace kmm