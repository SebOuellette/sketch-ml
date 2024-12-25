#version 460 core
precision highp float;
layout(local_size_x = 1) in;

float E = 2.71828182846;

struct Neuron {
    float bias;
    float value;
    float expected; // For non-final-layers, this value represents the 'output_delta' for the training session (only in backprop 2)
};

const uint TYPE_FULLY_CONNECTED = 0; // Fully conected layers are used in ANNs, and in stage 2 of CNNs.
const uint TYPE_CONVOLUTION = 1; // Convolutional layers are used in stage 1 of CNNs
const uint TYPE_POOLING = 2; // Pooling layers are used in stage 1 of CNNs
const uint TYPE_OUTPUT = 3; // The output layer of any network. Indicates no weights are allocated.

const uint POOLMETHOD_MAX = 0; // The maximum value found in the pooled input
const uint POOLMETHOD_AVG = 1; // The average value of the pooled input
const uint POOLMETHOD_MIN = 2; // The minimum value found in the pooled input

layout(std430, binding = 0) buffer ThisBuf {
    Neuron neurons[];
};

layout(std430, binding = 1) buffer OtherBuf {
    Neuron otherNeurons[];
};

layout(std430, binding = 2) buffer Weights {
    float weights[];
};

uniform bool isLastLayer;
uniform int nextCount;
uniform int thisCount;
uniform bool backProp;
uniform float learningRate;

// ==== G E N E R A L ====

// Soft step activation function
float activation(float x) {
    //return 1.0 / (1 + pow(E, -x));
    return 1.0 / (1 + exp(-x));
}

// The input to this is NOT 'x'. It is the result of sigmoid(x).
float activationD(float sigmoid) {
    // Force the network to always do a bit of learning by adding a bit of an offset to the activation derivitive (0.005)
    return (sigmoid * (1.f - sigmoid)) + 0.005; //0.005; //0.005;
}

// calculate the cost of a value and its expected
float valCost(float actual, float expected) {
    return pow(actual - expected, 1.0);
}

float valCostD(float actual, float expected) {
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
        weightIndex = windex(i, index); // Each larger block in weights is assocated with 'this' index
        newValue += weights[weightIndex] * neurons[i].value;
    }

    return float(newValue) + otherNeurons[index].bias;
}

void doForwardPass(uint index) {
    otherNeurons[index].value = activation(calcZ(index));
}

// Fully-Connected layer backpropagation
void doBackProp2(uint index) {
    // Output
    float thisActivationCost = 0.0;

    // Temp to be reused
    float error = 0.0;
    float delta = 0.0;
    uint weightIndex = 0;

    for (uint i = 0; i < nextCount; i++) {
        weightIndex = windex(index, i);

        if (isLastLayer) {
            // Calculate error and delta for last layer
            error = learningRate * valCostD(otherNeurons[i].value, otherNeurons[i].expected);
        } else {
            // Calculate error and delta for hidden layer(s)
            // In this case, 'expected' is actually the calculated activation cost sum from the next layer, calculated from the last backpropagation phase on that layer
            error = otherNeurons[i].expected;
        }

        // Calculate the activation derividive delta. We can use this for 3 things - adjusting weights, adjusting bias, and carrying backwards (using the derivitive of the last activation, which is the weight)
        delta = activationD(otherNeurons[i].value) * error;
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

void main() {
    uint index = gl_WorkGroupID.x; // This neuron index

    if (backProp) {
        // Index is the index in this layer
        doBackProp2(index);
    } else {
        // Index is the index in the next layer.
        doForwardPass(index);
    }
}
