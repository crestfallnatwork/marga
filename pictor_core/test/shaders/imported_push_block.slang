/* SPDX-License-Identifier: MPL-2.0 */

// Imported module declaring a public push constant. The entry file's program layout does not
// list it, so an entry that uses it must be rejected instead of silently binding void.

public struct ImportedIn {
    uint32_t* out;
    uint32_t  value;
};

[[vk::push_constant]]
public uniform ImportedIn imported;

public void write_imported()
{
    *imported.out = imported.value;
}
