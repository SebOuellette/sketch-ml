#version 460 core
precision highp float;
layout(local_size_x = 1) in;

const float E = 2.71828182846;
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
    float expected; // For non-final-layers, this value represents the 'output_delta' for the training session (only in backprop 2)
};

layout(std430, binding = 0) buffer ThisBuf {
    Neuron neurons[];
};

layout(std430, binding = 1) buffer OtherBuf {
    Neuron otherNeurons[];
};

layout(std430, binding = 2) buffer Weights {
    float weights[];
};

uniform int thisLayerType;
uniform int thisCount;
uniform int nextLayerType;
uniform int nextCount;
uniform ivec3 neuronDims;
uniform ivec3 weightDims;

uniform bool backProp;
uniform float learningRate;

// ==== G E N E R A L ====

// Soft step activation function
float sigmoid(float x) {
    //return 1.0 / (1 + pow(E, -x));
    return 1.0 / (1 + exp(-x));
}

// The input to this is NOT 'x'. It is the result of sigmoid(x).
float sigmoid_(float activation) {
    // Force the network to always do a bit of learning by adding a bit of an offset to the activation derivitive (0.005)
    return (activation * (1.f - activation)) + 0.005;
}

// calculate the cost of a value and its expected
float valCost(float actual, float expected) {
    return pow(actual - expected, 1.0);
}

float valCost_(float actual, float expected) {
    return 2.0 * (actual - expected);
}

// ==== F U L L Y - C O N N E C T E D ====

// @brief Calculate the Weight INDEX given the neuron index in layer A, and neuron index in layer B. Find the weight connecting the two neurons.
uint windex(uint neuronIndexLayerA, uint neuronIndexLayerB) {
    return neuronIndexLayerB * thisCount + neuronIndexLayerA;
}

float calcZ(uint index) {
    uint weightIndex = 0; // Weight index
    double newValue = 0;

    for (uint i = 0; i < thisCount; i++) {
        weightIndex = windex(index, i); // Each larger block in weights is assocated with 'this' index
        newValue += weights[weightIndex] * neurons[i].value;
    }

    return float(newValue) + otherNeurons[index].bias;
}

void fcForwardPass(uint index) {
    otherNeurons[index].value = sigmoid(calcZ(index));
}

// Fully-Connected layer backpropagation
void fcBackPropagate(uint index) {
    // Output
    float thisActivationCost = 0.0;

    // Temp to be reused
    float error = 0.0;
    float delta = 0.0;
    uint weightIndex = 0;

    for (uint i = 0; i < nextCount; i++) {
        weightIndex = windex(index, i);

        if (nextLayerType == TYPE_OUTPUT) {
            // Calculate error and delta for last layer
            error = learningRate * valCost_(otherNeurons[i].value, otherNeurons[i].expected);
        } else {
            // Calculate error and delta for hidden layer(s)
            // In this case, 'expected' is actually the calculated activation cost sum from the next layer, calculated from the last backpropagation phase on that layer
            error = otherNeurons[i].expected;
        }

        // Calculate the activation derividive delta. We can use this for 3 things - adjusting weights, adjusting bias, and carrying backwards (using the derivitive of the last activation, which is the weight)
        delta = sigmoid_(otherNeurons[i].value) * error;
        thisActivationCost += weights[weightIndex] * delta; // Carry over the weight before we adjust it
        weights[weightIndex] -= neurons[index].value * delta;

        // Only adjust biases for the last layer if we're index 0. All threads have the same delta in theory.. so they will all set to the same
        // I just want to syncrhonize so they don't corrupt or whatever
        if (index == 0) {
            otherNeurons[i].bias -= delta; // The derivitive of z with respect to b is 1.0
        }
    }

    // Carry the activation cost backwards
    // Carry 'output_delta' to the next (previous) layer
    neurons[index].expected = thisActivationCost;
}

// ==== C O N V O L U T I O N A L ====
uniform ivec3 filterSize;
uniform uint filterCount;

// Get the index for some layer or filter, based on the x,y,channels scale of the layer or filter, and the position of the item to index within.
uint getLayerIndex(ivec3 itemPos, ivec3 scale) {
    uint index = 0;
    index += itemPos.x;
    index += itemPos.y * scale.x;
    index += itemPos.z * scale.x * scale.y;

    return index;
}

// Translate a position relateive to the filter bottom left 0x0 origin, to an index around the input/output layer offset by the filter position
ivec3 translateFilterToNeuron(uvec3 neuronIndex, ivec3 filterPos) {
    ivec3 outputPos = ivec3(neuronIndex);

    // Offset the output Pos to have the filter position centered.
    outputPos += filterPos - filterPos / 2;

    return outputPos;
}

// Forward Propagation
void ConvolutionPropagate(uvec3 index) {
    float finalZSum = 0;

    // 'index' represents the index within the output layer, which is also the index within the input layer

    // Now do convolution

    // Add bias

    // Set activated neuron value
}

// Backpropagation
void ConvolutionBackpropagate(uvec3 index) {}

// ==== P O O L I N G ====
uniform ivec2 poolSize;
uniform int poolMethod;

// ==== A C T I O N   C O M P I L A T I O N =====
void doFullyConnected() {
    uint index = gl_WorkGroupID.x; // This neuron index

    if (backProp) {
        // Index is the index in this layer
        fcBackPropagate(index);
    } else {
        // Index is the index in the next layer.
        fcForwardPass(index);
    }
}

void doConvolution() {
    uvec3 index = gl_WorkGroupID; // This neuron index, also the channel.

    if (backProp) {
        ConvolutionBackpropagate(index);
    } else {
        ConvolutionPropagate(index);
    }
}

void doPooling() {
    if (backProp) {} else {}
}

void main() {
    // Do some action based on the layer type
    switch (thisLayerType) {
        case TYPE_FULLY_CONNECTED:
        {
            doFullyConnected();
            break;
        }

        case TYPE_CONVOLUTION:
        {
            doConvolution();
            break;
        }

        case TYPE_POOLING:
        {
            doPooling();
            break;
        }

        case TYPE_OUTPUT:
        default:
        {
            // No action
            break;
        }
    }
}
