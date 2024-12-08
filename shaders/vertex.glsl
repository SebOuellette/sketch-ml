#version 460 core
layout(location = 0) in vec3 aPos;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat4 rotation;

out vec3 FragPos;

void main() {
    vec4 newPos = vec4(aPos, 1.0);
    vec4 modeledPos = model * newPos;
    gl_Position = projection * view * modeledPos;
    FragPos = modeledPos.xyz;
}
