/* SPDX-License-Identifier: MPL-2.0 */

#pragma once

#include <span>
#include <string_view>

namespace pictor {

struct ShaderSourceFile {
    std::string_view path;
    std::string_view code;
};

struct ShaderSources {
    std::string_view entryPath;
    std::span<const std::string_view> searchPaths;
    std::span<const ShaderSourceFile> files;
};

} // namespace pictor
