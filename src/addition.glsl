#version 450

layout(local_size_x = 64) in;

layout(set = 0, binding = 0) buffer Left {
    float left[];
};
layout(set = 0, binding = 1) readonly buffer Right {
    float right[];
};
layout(set = 0, binding = 2) uniform Uniforms {
    uint size;
};

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i < size) {
        left[i] += right[i];
    }
}
