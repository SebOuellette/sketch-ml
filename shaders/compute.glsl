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

// Calculate the derivitive of the sigmoid function
float activationD(float x) {
    //return activation(x) * (1.0 - activation(x));
    return x * (1.0 - x);
}

// calculate the cost of a value and its expected
float valCost(float actual, float expected) {
    return pow(actual - expected, 1.0);
}

// Calculate the derivitive of a value and its expected
float valCostd(float actual, float expected) {
    return 1.0 * (actual - expected);
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

float calcBiasCost(uint index) {
    float activD = activationD(calcZ(index));
    float valcD = valCostd(neurons[index].value, neurons[index].expected);
    return activD * valcD;
}

float calcWeightCost(uint lastIndex, uint thisIndex) {
    float weightD = otherNeurons[lastIndex].value; // Derivitive of  weight * value + bias derived over weight
    return weightD * calcBiasCost(thisIndex); // We reuse the calcBiasCost to save code
}

void doForwardPass(uint index) {
    neurons[index].value = activation(calcZ(index));
}

void doBackProp(uint index) {
    // Update the expected values for the 'last' layer
    double weightCost = 0.0;
    double biasCost = 0.0;
    double valueCost = 0.0;

    uint weightIndex = 0; // Weight index

    for (uint i = 0; i < thisCount; i++) {
        weightIndex = i * lastCount + index; // Each larger block in weights is assocated with 'this' index

        //weightCost += calcWeightCost(index, i);
        //biasCost += calcBiasCost(i);
        float thisValCost = valCost(neurons[i].value, neurons[i].expected);
        valueCost += thisValCost;

        weights[weightIndex] -= learningRate * 1.0 * thisValCost / thisCount;
    }

    //weightCost /= lastCount;
    //biasCost /= lastCount;

    //neurons[index].bias -= learningRate * float(valueCost) / thisCount;
    //weights[index] -= float(valueCost) / thisCount;
    otherNeurons[index].expected = otherNeurons[index].value - float(valueCost) / thisCount;

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
