#pragma once

#include <cstdint>
#include <cstring>

namespace pictor::bench {

// Deterministic synthetic data shared by the CUDA and Vulkan benchmark binaries, so both scan the
// same logical sequence (and each can verify against the same host reference).
inline uint64_t splitmix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30u)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27u)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31u);
}

template <typename T> inline T synthetic_value(uint64_t i)
{
    return static_cast<T>(splitmix64(i + 0x5DEECE66Dull));
}

// Reinterprets the synthetic bits as `T`, preserving the bit pattern for float/double keys.
template <typename T> inline T synthetic_at(uint64_t i)
{
    T v;
    if constexpr (sizeof(T) == 8) {
        const auto bits = synthetic_value<uint64_t>(i);
        std::memcpy(&v, &bits, sizeof(T));
    } else {
        const auto bits = synthetic_value<uint32_t>(i);
        std::memcpy(&v, &bits, sizeof(T));
    }
    return v;
}

// The bit pattern of a 32- or 64-bit value.
template <typename T> inline uint64_t bits_of(const T& v)
{
    if constexpr (sizeof(T) == 8) {
        uint64_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return bits;
    } else {
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return bits;
    }
}

// Streaming host reference for exclusive/inclusive scan over the synthetic sequence.
template <typename T> class StreamScanReference {
  public:
    explicit StreamScanReference(bool exclusive) : _exclusive(exclusive) {}

    T expected(T input)
    {
        T out = _exclusive ? _acc : static_cast<T>(_acc + input);
        _acc  = static_cast<T>(_acc + input);
        return out;
    }

  private:
    bool _exclusive;
    T _acc{0};
};

} // namespace pictor::bench
