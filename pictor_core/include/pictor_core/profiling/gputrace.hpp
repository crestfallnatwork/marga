/* SPDX-License-Identifier: MPL-2.0 */

#pragma once

#include <vulkan/vulkan.hpp>

namespace pictor::gputrace {

bool initialize(vk::Queue queue = {});
[[nodiscard]] bool available();

void start();
void stop();

class ScopedRange {
  public:
    ScopedRange()
    {
        start();
    }

    ~ScopedRange()
    {
        stop();
    }

    ScopedRange(const ScopedRange&)            = delete;
    ScopedRange& operator=(const ScopedRange&) = delete;
};

} // namespace pictor::gputrace
