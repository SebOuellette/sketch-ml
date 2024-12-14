#version 460 core

in vec3 FragPos;
in vec4 vertCol;

out vec4 FragColor;

uniform vec2 resolution;
uniform vec2 layerSize;
uniform vec2 cursor;
uniform bool leftClick;
uniform bool rightClick;
uniform bool lalt;
uniform bool lctrl;
uniform float drawSize;

uniform vec3 screenPos;
uniform vec3 screenSize;

struct Neuron {
    float bias;
    float value;
    float expected;
};

layout(std430, binding = 0) buffer SSBO {
    Neuron neurons[];
};

void main() {
    //vec2 uv = vec2(max(-resolution.x / 2.0, gl_FragCoord.x), max(-resolution.y / 2.0, gl_FragCoord.y)) / min(resolution.x, resolution.y) - FragPos.xy;
    vec2 uv = (FragPos.xy - screenPos.xy + (screenSize.xy / 2)) / screenSize.xy;
    int index = int(mod((floor(uv.y * layerSize.x) + uv.x) * layerSize.x, layerSize.x * layerSize.y));

    float cursorDist = distance(gl_FragCoord.xy, vec2(cursor.x, resolution.y - cursor.y));
    bool mouseRange = cursorDist < drawSize;

    if (int(cursorDist) == int(drawSize)) {
        FragColor = vec4(1.0, 0.0, 0.0, 1.0);
        return;
    }

    // User input
    float val;
    float minV = 0.0;

    if (mouseRange) {
        if (lalt) {
            val = neurons[index].expected;
            if (!lctrl) {
                minV = -1.0;
            }
        } else {
            val = neurons[index].value;
        }

        if (leftClick) {
            val = lalt ? 1.0 : min(val + 0.2, 1.0);
        } else if (rightClick) {
            val = lalt ? -1.0 : max(val - 0.2, minV);
            if (lctrl) {
                val = 0.0;
            }
        }

        //val = min(max(val, minV), 1.0);

        if (lalt) {
            neurons[index].expected = val;
        } else {
            neurons[index].value = val;
        }
    }

    // Draw
    float cost = neurons[index].expected - neurons[index].value;
    //FragColor = vec4(vec3(neurons[index].bias, 0, neurons[index].expected), 1.0);
    //FragColor = vec4(vec3(abs(cost), 0, 0), 1.0);
    FragColor = vec4(vec3(neurons[index].value) + vec3(-neurons[index].expected, neurons[index].bias * 0.2, neurons[index].expected) * 0.3, 1.0);
}
