#pragma once

#include "kmm/core/copy_def.hpp"

namespace kmm {

void execute_copy(const void* src_buffer, void* dst_buffer, CopyDef copy_description);

}