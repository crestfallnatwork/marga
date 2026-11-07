/* SPDX-License-Identifier: MPL-2.0 */

public uint wave_mix(uint v)
{
    return v * 3u + 1u;
}

public uint wave_broadcast(uint v)
{
    return WaveReadLaneAt(v, 0);
}
