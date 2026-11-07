/* SPDX-License-Identifier: MPL-2.0 */

// Second translation unit of the generated-header link test; see tunables_multi_tu_a.cpp.

#include "tunable_test.shader.h"

#include <cstdint>

bool tunables_multi_tu_a();

bool tunables_multi_tu_b()
{
    tunable_test_shader::TunableTestTunables tunables;
    return tunables.defines().view().size() == 5;
}

int main()
{
    return (tunables_multi_tu_a() && tunables_multi_tu_b()) ? 0 : 1;
}
