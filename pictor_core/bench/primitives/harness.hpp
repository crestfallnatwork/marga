/* SPDX-License-Identifier: MPL-2.0 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace pictor::bench {

// Splits a comma-separated list, dropping empty entries.
inline std::vector<std::string> split(const std::string& s)
{
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t comma = s.find(',', start);
        if (comma == std::string::npos)
            comma = s.size();
        if (comma > start)
            out.push_back(s.substr(start, comma - start));
        start = comma + 1;
    }
    return out;
}

// Middle sample of the per-iteration timings.
inline double median(std::vector<double> v)
{
    std::ranges::sort(v);
    return v[v.size() / 2];
}

// Maps `bits` onto the order-preserving key the radix sorts implement: `order` is 0 unsigned,
// 1 signed, 2 float, and `key_bits` is the key width.
inline uint64_t order_key(uint64_t bits, uint32_t key_bits, uint32_t order)
{
    const uint64_t sign = 1ull << (key_bits - 1);
    switch (order) {
        case 1:
            return bits ^ sign;
        case 2:
            return (bits & sign) != 0 ? ~bits : (bits | sign);
        default:
            return bits;
    }
}

// Walks argv[1..], consuming flags and their values. A flag with a missing value and an unknown
// flag exit(1) with the messages the harnesses printed before.
class ArgReader {
  public:
    ArgReader(int argc, char** argv) : _argv(argv), _argc(argc) {}

    bool next(std::string& out)
    {
        if (_index >= _argc)
            return false;
        out = _argv[_index++];
        return true;
    }

    std::string value(const char* name)
    {
        if (_index >= _argc) {
            std::fprintf(stderr, "missing value for %s\n", name);
            std::exit(1);
        }
        return _argv[_index++];
    }

    [[noreturn]] void unknown(const std::string& k) const
    {
        std::fprintf(stderr, "unknown option %s\n", k.c_str());
        std::exit(1);
    }

  private:
    char** _argv;
    int _argc;
    int _index = 1;
};

} // namespace pictor::bench
