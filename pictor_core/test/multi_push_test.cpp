/* SPDX-License-Identifier: MPL-2.0 */

// Compile-time proof that pictor_reflect bound each executor to the push-constant block its
// entry actually uses. multi_push declares three blocks before the entries; same_type declares
// two blocks of one type.

#include "multi_push.shader.h"
#include "same_type_push.shader.h"

#include <type_traits>

static_assert(std::is_same_v<multi_push_shader::AlphaComputeExecutor::ParamsType,
                             multi_push_shader::AlphaIn>);
static_assert(
    std::is_same_v<multi_push_shader::BetaComputeExecutor::ParamsType, multi_push_shader::BetaIn>);
static_assert(
    std::is_same_v<multi_push_shader::GAMMAExecutor::ParamsType, multi_push_shader::GammaIn>);

static_assert(std::is_same_v<same_type_push_shader::FirstComputeExecutor::ParamsType,
                             same_type_push_shader::SharedIn>);
static_assert(std::is_same_v<same_type_push_shader::SecondComputeExecutor::ParamsType,
                             same_type_push_shader::SharedIn>);
static_assert(std::is_same_v<same_type_push_shader::BothComputeExecutor::ParamsType,
                             same_type_push_shader::SharedIn>);

int main()
{
    return 0;
}
