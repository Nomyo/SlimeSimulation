#version 450

layout (binding = 0) uniform sampler2D samplerColorMap;
layout (location = 0) in vec2 inUV;
layout (location = 1) in vec4 inColor;
layout (location = 0) out vec4 outFragColor;

void main ()
{
    outFragColor = texture(samplerColorMap, inUV);
}