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
uniform uvec3 neuronDims;
uniform uvec3 weightDims;

uniform bool backProp;
uniform float learningRate;

// ==== G E N E R A L ====

// === a c t i v a t i o n   f u n c t i o n s ===
// @brief Soft step activation function
// @param[in] x	The 'x' value of the equation. AKA time, or whatever else.
// @return		The result of the sigmoid activation function given 'x'
float sigmoid(float x) {
    //return 1.0 / (1 + pow(E, -x));
    return 1.0 / (1 + exp(-x));
}

// @brief 		The input to this is NOT 'x'. It is the result of sigmoid(x).
// @details		This is for efficiency reasons, why recalculate on backpropagation when we can just use the result from the last propagation?
// @param[in] activation	The result from a sigmoid activation function. This is NOT 'x'.
// @return 					The derivitive of the provided activation function.
float sigmoid_(float activation) {
    // Force the network to always do a bit of learning by adding a bit of an offset to the activation derivitive (0.005)
    return (activation * (1.f - activation)) + 0.005;
}

// @brief Leaky ReLU
// @param[in] x	The 'x' value or time provided to the ReLU function
// @return		The ReLU result
float lrelu(float x) {
    return max(x, learningRate * x);
}

// @brief leaky ReLU derivitive. The derivitive is the same whether you pass it x or the activation of lrelu :)
// @param[in] activation	The activation result from a leaky ReLU function
// @return					The derivitive of the leaky relu
float lrelu_(float activation) {
    return (activation > 0) ? 1.0 : learningRate;
}

// === c o s t   f u n c t i o n s ===
// @brief Calculate mean square error of the an actual and expected neurons
// @param[in] actual	The actual calculated neuron value
// @param[in] expected	The expected neuron value
// @return				The cost using mean square error
float valCost(float actual, float expected) {
    return pow(actual - expected, 2.0);
}

// @brief The derivitive of the mean square error
// @param[in] actual	The actual calculated neuron value
// @param[in] expected	The expected neuron value
// @return				The derivitive of the cost using mean square error
float valCost_(float actual, float expected) {
    return 2.0 * (actual - expected);
}

// ==== F U L L Y - C O N N E C T E D ====
// @brief Calculate the Weight INDEX given the neuron index in layer A, and neuron index in layer B. Find the weight connecting the two neurons.
// @param[in] neuronIndexLayerA	The index of this layer
// @param[in] neuronIndexLayerB	The index of the next layer
// @return						The index of the weight between a fully connected layer A and layer B
uint windex(uint neuronIndexLayerA, uint neuronIndexLayerB) {
    return neuronIndexLayerB * thisCount + neuronIndexLayerA;
}

// @brief Perform fully-connected propagation, without activation
// @param[in] index	The index of the neuron in the next layer
// @return			The result from propagation without activation
float calcZ(uint index) {
    uint weightIndex = 0; // Weight index
    double newValue = 0;

    for (uint i = 0; i < thisCount; i++) {
        weightIndex = windex(i, index); // Each larger block in weights is assocated with 'this' index
        newValue += weights[weightIndex] * neurons[i].value;
    }

    return float(newValue) + otherNeurons[index].bias;
}

// @brief Perform activation and set the result for fully-connected propagation
// @param[in] index	The index of the neuron in the next layer
void fcForwardPass(uint index) {
    otherNeurons[index].value = sigmoid(calcZ(index));
}

// @brief Fully-Connected layer backpropagation
// @param[in] index	The index od the neuron in the next layer
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
uniform uvec3 filterSize;
uniform uint filterCount;

// @brief Get the index for some layer or filter, based on the x,y,channels scale of the layer or filter, and the position of the item to index within.
// @details The 'w' component of itemPos represents the filter index, which filter out of the list of filters in this layer is this one. THIS IS 0 FOR NEURON/BIAS/EXPECTED LAYERS
// @param[in] itemPos	The position of the filter position index or neuron index. The w component is only used to represent the whole filter index.
// @param[in] scale		The scale of the layer, either a single filter block or full neuron layer.
// @return				The index within this block for use in a single dimensional array
uint getLayerIndex(ivec4 itemPos, ivec3 scale) {
    uint index = 0;
    uint multiplier = 1;
    index += itemPos.x;
    multiplier *= scale.x;
    index += itemPos.y * multiplier;
    multiplier *= scale.y;
    index += itemPos.z * multiplier;
    multiplier *= scale.z;
    index += itemPos.w * multiplier;

    return index;
}

// @brief Translate a position relateive to the filter bottom left 0x0 origin, to an index around the input/output layer offset by the filter position
// @param[in] neuronIndex	The index of the neuron in the next layer. Only x,y is used, as this calculation does not take multiple filters into account. This is just position within a single filter
// @param[in] filterPos		The position of the weight within the filter to offset for
// @return					The position of the neuron to multiply by the weight in the filter
ivec3 translateFilterToNeuron(uvec2 neuronIndex, ivec3 filterPos) {
    ivec3 outputPos = ivec3(neuronIndex);

    // Offset the output Pos to have the filter position centered.
    outputPos.xy += filterPos.xy - filterPos.xy / 2;

    // The z position is not offset. Each filter's channel corresponds with one input channel. No mixing and matching.
    // But, for forward propagation.. it just makes more sense to copy filterPos.z to the output. The filter dimensions contain all information we need about depth for input and output layers.
    outputPos.z = filterPos.z;

    return outputPos;
}

// @brief Forward Propagation
// @param[in] index	The index of the neuron in the next layer to use for propagation
void ConvolutionPropagate(uvec3 index) {
    float finalZSum = 0;

    // 'index' represents the index within the output layer, which is also the index within the input layer
    uint nextNeuronIndex = getLayerIndex(ivec4(index, 0), nextCount);

    // temp vars
    ivec3 filterPos;
    ivec3 newPos;
    uint neuronIndex;
    uint filterIndex;

    // Now do convolution
    /// Loop through each filter.
    /// Multiply filter index by the corresponding input neuron, add result to finalZSum
    for (uint filterZ = 0; filterZ < filterSize.z; filterZ++) {
        for (uint filterY = 0; filterY < filterSize.y; filterY++) {
            for (uint filterX = 0; filterX < filtsrSize.x; filterX++) {
                // GLSL automatically pads with 0 if it's out of range.. but even if the texture looped.. that's another acceptable way of doing this.
                filterPos = ivec3(filterX, filterY, filterZ);

                // Fetch the position of the neuron to multiply by the filter value
                newPos = translateFilterToNeuron(index, filterPos);

                neuronIndex = getLayerIndex(ivec4(newPos, 0), nextCount); // Even though we're finding an index on thisLayer... we use nextLayer's size. They *should* match under all cases.. but if they don't, it's just safer to use nextCount in my brain
                filterIndex = getLayerIndex(ivec4(filterPos, filterCount), filterSize);

                // Now that we have the indices, we can multiply the values at those indices in the respective arrays
                finalZSum += weights[filterIndex] * neurons[neuronIndex];
            }
        }
    }

    // Add bias to finalZSum
    finalZSum += otherNeurons[nextNeuronIndex].bias;

    // pass finalZSum through activation (RELU)
    otherNeurons[nextNeuronIndex].value =
}

// @brief Backpropagation
// @param[in] index	The index of the neuron in this layer to use for propagation
void ConvolutionBackpropagate(uvec3 index) {}

// ==== P O O L I N G ====
uniform ivec2 poolSize;
uniform int poolMethod;

// ==== A C T I O N   C O M P I L A T I O N =====
// @brief Perform fully connected propagation or backpropagation
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

// @brief Perform convolution propagation or backpropagation
void doConvolution() {
    uvec3 index = gl_WorkGroupID; // This neuron index, also the channel.

    if (backProp) {
        ConvolutionBackpropagate(index);
    } else {
        ConvolutionPropagate(index);
    }
}

// @brief Perform pooling
void doPooling() {
    if (backProp) {} else {}
}

// @brief Perform network actions.
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
