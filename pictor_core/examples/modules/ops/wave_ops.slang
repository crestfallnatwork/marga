/* SPDX-License-Identifier: MPL-2.0 */

// Modules can import other modules. This import is relative to this file, so it
// resolves to ops/scale.slang; pictor_reflect embeds the whole transitive closure.
import "scale";

public uint wave_mix(uint v)
{
    return scale_value(v);
}

// Wave intrinsics are available in any module; WAVE_SIZE comes from the runtime
// subgroup size when the executor compiles the closure.
public uint wave_broadcast(uint v)
{
    return WaveReadLaneAt(v, 0);
}
