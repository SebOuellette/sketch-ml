#version 460 core
precision highp float;
layout(local_size_x = 1) in;

float E = 2.71828182846;

struct Neuron {
    float bias;
    float value;
    float expected;
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

uniform int lastCount;
uniform int thisCount;
uniform bool backProp;
uniform float learningRate;

// Soft step activation function
float activation(float x) {
    //return 1.0 / (1 + pow(E, -x));
    return 1.0 / (1 + exp(-x));
}

// calculate the cost of a value and its expected
float valCost(float actual, float expected) {
    return pow(actual - expected, 1.0);
}

float calcZ(uint index) {
    uint weightIndex = 0; // Weight index

    double newValue = 0;

    for (uint i = 0; i < lastCount; i++) {
        weightIndex = index * lastCount + i; // Each larger block in weights is assocated with 'this' index
        newValue += weights[weightIndex] * otherNeurons[i].value;
    }

    return float(newValue) + neurons[index].bias;
}

void doForwardPass(uint index) {
    neurons[index].value = activation(calcZ(index));
}

void doBackProp(uint index) {
    // Update the expected values for the 'last' layer
    double valueCost = 0.0;

    uint weightIndex = 0; // Weight index

    for (uint i = 0; i < thisCount; i++) {
        weightIndex = i * lastCount + index; // Each larger block in weights is assocated with 'this' index

        float thisValCost = valCost(neurons[i].value, neurons[i].expected);
        valueCost += thisValCost;

        weights[weightIndex] -= learningRate * otherNeurons[index].value * thisValCost; // THIS WORKED WITH MOST NUMBERS ???
    }

    valueCost /= thisCount;

    neurons[index].bias -= learningRate * float(valueCost);
    otherNeurons[index].expected = otherNeurons[index].value - float(valueCost) / 20.0; // dividing by 10 creates a batch of 10.. I think.. and it works? soo uhhh ? Why does everyone need calculus? It's just intuitive ratios.

    // What?
}

void main() {
    uint index = gl_WorkGroupID.x; // This neuron index

    if (backProp) {
        doBackProp(index);
    } else {
        doForwardPass(index);
    }
}
