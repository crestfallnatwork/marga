#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 64) in;
layout(max_vertices = 1, max_primitives = 1) out;
layout(points) out;

layout(buffer_reference, std430) buffer FloatArray {
    float data[];
};

layout(push_constant) uniform PushConstants {
    uint64_t a_addr;
    uint64_t b_addr;
    uint64_t c_addr;
    uint count;
} pc;

void main() {
    uint gId = gl_GlobalInvocationID.x;
    if (gId < pc.count) {
        FloatArray a = FloatArray(pc.a_addr);
        FloatArray b = FloatArray(pc.b_addr);
        FloatArray c = FloatArray(pc.c_addr);
        c.data[gId] = a.data[gId] + b.data[gId];
    }
}
