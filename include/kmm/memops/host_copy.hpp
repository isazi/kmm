#pragma once

#include "kmm/core/copy_description.hpp"

namespace kmm {

void execute_copy(const void* src_buffer, void* dst_buffer, CopyDescription copy_description);

}