#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 1) in;

layout(buffer_reference, std430) buffer FloatArray {
    float data[];
};

layout(push_constant) uniform PushConstants {
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint count;
} pc;

void main() {
    uint numGroups = (pc.count + 63) / 64;
    EmitMeshTasksEXT(numGroups, 1, 1);
}
