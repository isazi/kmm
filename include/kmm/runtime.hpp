#pragma once
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "kmm/types.hpp"

namespace kmm {

class Runtime {
  public:
  private:
    std::shared_ptr<RuntimeImpl> m_impl;
};

}  // namespace kmm