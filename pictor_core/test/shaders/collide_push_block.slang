/* SPDX-License-Identifier: MPL-2.0 */

// Imported block with the same variable name as an entry-file block but a different type. The
// type must win; matching by variable name alone would silently bind LocalIn.

public struct ImportedParams {
    uint32_t* out;
    uint32_t  value;
};

[[vk::push_constant]]
public uniform ImportedParams shared;

public void write_shared()
{
    *shared.out = shared.value;
}
