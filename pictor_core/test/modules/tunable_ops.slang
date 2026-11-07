/* SPDX-License-Identifier: MPL-2.0 */

// A tunable declared in a module: pictor_reflect scans the whole dependency closure, so
// PICTOR_MODULE_BIAS is extracted into the entry shader's generated Tunables struct even
// though it is declared here.

#ifndef PICTOR_MODULE_BIAS
#define PICTOR_MODULE_BIAS 10
#endif

public uint module_bias()
{
    return PICTOR_MODULE_BIAS;
}
