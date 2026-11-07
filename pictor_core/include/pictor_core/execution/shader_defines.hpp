/* SPDX-License-Identifier: MPL-2.0 */

#pragma once

#include <deque>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace pictor {

struct ShaderDefine {
    std::string_view name;
    std::string_view value;
};

struct CompileOptions {
    std::span<const ShaderDefine> defines;
};

class ShaderDefines {
  public:
    ShaderDefines()                                    = default;
    ShaderDefines(const ShaderDefines&)                = delete;
    ShaderDefines& operator=(const ShaderDefines&)     = delete;
    ShaderDefines(ShaderDefines&&) noexcept            = default;
    ShaderDefines& operator=(ShaderDefines&&) noexcept = default;

    // A later add() may reallocate _defs and invalidate any view() returned before it.
    void add(std::string_view name, std::string_view value)
    {
        _storage.emplace_back(name);
        _storage.emplace_back(value);
        _defs.push_back(ShaderDefine{_storage[_storage.size() - 2], _storage[_storage.size() - 1]});
    }

    [[nodiscard]] std::span<const ShaderDefine> view() const
    {
        return _defs;
    }

  private:
    std::deque<std::string> _storage;
    std::vector<ShaderDefine> _defs;
};

} // namespace pictor
