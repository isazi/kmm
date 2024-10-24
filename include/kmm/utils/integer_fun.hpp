#pragma once

#include <limits>

namespace kmm {

/**
 * Divide `num` by `denom` and round the result down.
 */
template<typename T>
T div_floor(T a, T b) {
    T zero = static_cast<T>(0);
    T quotient = a / b;

    // Adjust the quotient if a and b have different signs
    if (a % b != zero && ((a >= zero) ^ (b >= zero))) {
        quotient -= 1;
    }

    return quotient;
}

/**
 * Divide `num` by `denom` and round the result up.
 */
template<typename T>
T div_ceil(T a, T b) {
    T zero = static_cast<T>(0);

    if (b == zero) {
        return zero;
    }

    T quotient = a / b;

    // Adjust the quotient if both a and b have the same sign
    if (a % b != zero && !((a >= zero) ^ (b >= zero))) {
        quotient += 1;
    }

    return quotient;
}

/**
 * Round `input` to the first multiple of `multiple`.
 *
 * In other words, returns the smallest value not less than `input` that is divisible by `multiple`.
 */
template<typename T>
T round_up_to_multiple(T input, T multiple) {
    T zero = static_cast<T>(0);

    if (multiple == zero) {
        return input;
    }

    if (multiple < zero) {
        multiple = -multiple;
    }

    T remainder = input % multiple;

    if (remainder == zero) {
        return input;
    } else if (input < zero) {
        return input - remainder;
    } else {
        return input + (multiple - remainder);
    }
}

/**
 * Return the smallest number that is a power of two and is not less than `input`. This function
 * returns `numeric_limits<T>::max()` if not such number exists.
 */
template<typename T>
static T round_up_to_power_of_two(T input) {
    if (input <= static_cast<T>(0)) {
        return static_cast<T>(1);
    }

    input -= static_cast<T>(1);
    for (size_t i = 1; i < sizeof(T) * 8; i *= 2) {
        input |= (input >> i);
    }

    // What to do with overflows?
    if (input == std::numeric_limits<T>::max()) {
        return std::numeric_limits<T>::max();
    }

    input += static_cast<T>(1);
    return input;
}

/**
 * Check if the given number is a power of two.
 */
template<typename T>
static bool is_power_of_two(T input) {
    if (input <= static_cast<T>(0)) {
        return false;
    }

    return (input & (input - 1)) == static_cast<T>(0);
}

}  // namespace kmm