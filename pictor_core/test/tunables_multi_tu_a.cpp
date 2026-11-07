/* SPDX-License-Identifier: MPL-2.0 */

// One of two translation units that both include the generated tunables header and
// ODR-use TunableTestTunables::defines(). The generated definition must be inline or the
// two TUs collide at link time.

#include "tunable_test.shader.h"

#include <cstdint>

bool tunables_multi_tu_a()
{
    tunable_test_shader::TunableTestTunables tunables;
    return tunables.defines().view().size() == 5;
}
