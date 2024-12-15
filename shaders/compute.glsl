#version 460 core
precision highp float;
layout(local_size_x = 1) in;

float E = 2.71828182846;

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

uniform bool isLastLayer;
uniform int lastCount;
uniform int thisCount;
uniform bool backProp;
uniform float learningRate;

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

uint windex(uint lastIndex, uint thisIndex) {
    return thisIndex * lastCount + lastIndex;
}

void doBackProp1(uint index) {
    // Update the expected values for the 'last' layer
    double valueCost = 0.0; // total cost sum of all neurons between their expected values
    uint weightIndex = 0; // Store the weight index
    float thisValCost = 0.0; // Store the value cost for a single neuron

    for (uint i = 0; i < thisCount; i++) {
        // Each larger block in weights is assocated with 'this' index
        // A 'block' is a list of listCount floats, in which there are thisCount number of blocks. Therefore this is how we iterate given 'index' and 'i'
        weightIndex = windex(index, i);

        thisValCost = valCost(neurons[i].value, neurons[i].expected);
        valueCost += thisValCost;

        weights[weightIndex] -= learningRate * otherNeurons[index].value * thisValCost; // THIS WORKED WITH MOST NUMBERS ???
    }

    valueCost /= thisCount;

    neurons[index].bias -= learningRate * float(valueCost);
    otherNeurons[index].expected = otherNeurons[index].value - float(valueCost) / 20.0; // dividing by 10 creates a batch of 10.. I think.. and it works? soo uhhh ? Why does everyone need calculus? It's just intuitive ratios. 5 is too low. 20 is good, 10 is good too.

    // What?
}

void doBackProp2(uint index) {
    // Output
    float thisActivationCost = 0.0;

    // Temp to be reused
    float error = 0.0;
    float delta = 0.0;
    uint weightIndex = 0;

    for (uint i = 0; i < thisCount; i++) {
        weightIndex = windex(index, i);

        if (isLastLayer) {
            // Calculate error and delta for last layer
            error = learningRate * valCostD(neurons[i].value, neurons[i].expected);
        } else {
            // Calculate error and delta for hidden layer(s)
            // In this case, 'expected' is actually the calculated activation cost sum from the next layer, calculated from the last backpropagation phase on that layer
            error = neurons[i].expected;
        }

        // Calculate the activation derividive delta. We can use this for 3 things - adjusting weights, adjusting bias, and carrying backwards (using the derivitive of the last activation, which is the weight)
        delta = activationD(neurons[i].value) * error;
        thisActivationCost += weights[weightIndex] * delta; // Carry over the weight before we adjust it
        weights[weightIndex] -= otherNeurons[index].value * delta;

        // Only adjust biases for the last layer if we're index 0. All threads have the same delta in theory.. so they will all set to the same
        // I just want to syncrhonize so they don't corrupt or whatever
        if (index == 0) {
            neurons[i].bias -= delta; // The derivitive of z with respect to b is 1.0
        }
    }

    // Carry the activation cost backwards
    // Carry 'output_delta' to the next (previous) layer
    otherNeurons[index].expected = thisActivationCost;
}

void main() {
    uint index = gl_WorkGroupID.x; // This neuron index

    if (backProp) {
        doBackProp2(index);
    } else {
        doForwardPass(index);
    }
}
