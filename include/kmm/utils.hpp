#pragma once

#include <variant>
#include <vector>

namespace kmm {

template<typename T>
void remove_duplicates(T& input) {
    std::sort(std::begin(input), std::end(input));
    auto last_unique = std::unique(std::begin(input), std::end(input));
    input.erase(last_unique, std::end(input));
}

namespace detail {

template<typename... F>
struct overload: F... {
    explicit constexpr overload(F... f) : F(f)... {}
    using F::operator()...;
};

template<class... F>
overload(F...) -> overload<F...>;
}  // namespace detail

template<typename Variant, typename... Arms>
decltype(auto) match(Variant&& v, Arms&&... arms) {
    return std::visit(detail::overload {std::forward<Arms>(arms)...}, std::forward<Variant>(v));
}

}  // namespace kmm