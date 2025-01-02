#pragma once

#include "kmm/memops/types.hpp"

namespace kmm {

void execute_copy(const void* src_buffer, void* dst_buffer, CopyDef copy_def);

}