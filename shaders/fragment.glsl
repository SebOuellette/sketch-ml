#version 460 core

in vec3 FragPos;
in vec4 vertCol;

out vec4 FragColor;

uniform uint layerType;
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

const uint TYPE_FULLY_CONNECTED = 0; // Fully conected layers are used in ANNs, and in stage 2 of CNNs.
const uint TYPE_CONVOLUTION = 1; // Convolutional layers are used in stage 1 of CNNs
const uint TYPE_POOLING = 2; // Pooling layers are used in stage 1 of CNNs
const uint TYPE_OUTPUT = 3; // The output layer of any network. Indicates no weights are allocated.
const uint POOLMETHOD_MAX = 0; // The maximum value found in the pooled input
const uint POOLMETHOD_AVG = 1; // The average value of the pooled input
const uint POOLMETHOD_MIN = 2; // The minimum value found in the pooled input

struct Neuron {
    float bias;
    float value;
    float expected;
};

layout(std430, binding = 0) buffer SSBO {
    Neuron neurons[];
};

void main() {
    vec2 uv = (FragPos.xy - screenPos.xy + (screenSize.xy / 2)) / screenSize.xy;
    int index = int(mod((floor(uv.y * layerSize.x) + uv.x) * layerSize.x, layerSize.x * layerSize.y));

    float cursorDist = distance(gl_FragCoord.xy, vec2(cursor.x, resolution.y - cursor.y));
    bool mouseRange = cursorDist < drawSize;

    // This is the edge of the draw radius. Display a red outline.
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
    //if (layerType == TYPE_POOLING) {
    //  FragColor = vec4(vec3());
    //} else {
    FragColor = vec4(vec3(neurons[index].value) + vec3(-neurons[index].expected, 0.0, neurons[index].expected) * 30.3, 1.0);
    //}
}
