#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "fmt/format.h"

#include "kmm/core/config.hpp"
#include "kmm/utils/checked_math.hpp"

namespace kmm {

static size_t parse_byte_size(const char* input) {
    char* end;
    long double result = strtold(input, &end);

    if (end == input) {
        throw std::runtime_error(fmt::format("invalid size in bytes: {}", input));
    }

    auto unit = std::string(end);
    std::transform(unit.begin(), unit.end(), unit.begin(), ::tolower);

    if (unit.empty() || unit == "b") {
        //
    } else if (unit == "k" || unit == "kb") {
        result *= 1000;
    } else if (unit == "m" || unit == "mb") {
        result *= 1000'0000;
    } else if (unit == "g" || unit == "gb") {
        result *= 1000'000'000;
    } else if (unit == "t" || unit == "tb") {
        result *= 1000'000'000'000;
    } else {
        throw std::runtime_error(fmt::format("invalid size in bytes: {}", input));
    }

    if (result < 0 || result >= std::numeric_limits<size_t>::max()) {
        throw std::runtime_error(fmt::format("invalid size in bytes: {}", input));
    }

    return static_cast<size_t>(result);
}

WorkerConfig default_config_from_environment() {
    WorkerConfig config;

    if (auto* s = getenv("KMM_HOST_MEM")) {
        config.host_memory_limit = parse_byte_size(s);
    }

    if (auto* s = getenv("KMM_HOST_BLOCK")) {
        config.host_memory_block_size = parse_byte_size(s);
    }

    if (auto* s = getenv("KMM_DEVICE_MEM")) {
        config.device_memory_limit = parse_byte_size(s);
    }

    if (auto* s = getenv("KMM_DEVICE_BLOCK")) {
        config.device_memory_block_size = parse_byte_size(s);
    }

    return config;
}

}  // namespace kmm