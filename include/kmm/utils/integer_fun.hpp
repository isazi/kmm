#include <limits>

namespace kmm {

/**
 * Round `input` up such that it is a multiple of `multiple`. For example,
 * `round_up_to_multiple(13, 5)` returns `15`.
 */
template<typename T>
T round_up_to_multiple(T input, T multiple) {
    // Handle zero multiple to avoid division by zero
    if (multiple == T(0)) {
        return input;
    }

    if (multiple < T(0)) {
        multiple = -multiple;
    }

    T remainder = input % multiple;
    if (remainder == T(0)) {
        return input;
    }

    T diff;

    if (input < T(0)) {
        // For negative numbers, rounding up means subtracting the remainder
        return input - remainder;
    } else {
        // For positive numbers, rounding up means adding the difference
        return input + (multiple - remainder);
    }
}

/**
 * Return `input` up such that it is a positive power of 2. This function returns `0` if rounding
 * up results in an overflow. For negative numbers, this function returns `1`.
 */
template<typename T>
static T round_up_to_power_of_two(T input) {
    T result = T(1);

    while (input > result) {
        if (result > std::numeric_limits<T>::max() / 2) {
            return 0;
        }

        result *= T(2);
    }

    return result;
}

/**
 * Divide `num` by `denom` and round the result up.
 */
template<typename T>
T div_ceil(T num, T denom) {
    // Handle zero multiple to avoid division by zero
    if (denom == T(0)) {
        return T(0);
    }

    if (denom < T(0)) {
        denom = -denom;
    }

    T remainder = num % denom;
    return num / denom + ((num < T(0) || remainder == T(0)) ? T(0) : T(1));
}

/**
 * Check if the given number is a power of two.
 */
template<typename T>
static bool is_power_of_two(T input) {
    return round_up_to_power_of_two(input) == input;
}

}  // namespace kmm