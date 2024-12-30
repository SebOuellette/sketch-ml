#include "layers/conv_layer.h"

/* @brief The layer to copy. Just copies internal variables (Should also dereference stuff that is not needed anymore)
 * @param[in] copyLayer	A reference to the layer to copy
*/
ConvLayer::ConvLayer(ConvLayer const& copyLayer) {
	this->Layer::operator=(copyLayer);
}

/* @brief Setup a convolutional layer
 * @param[in] newNeuronDims	The new size of the neurons in 3 dimensional space
 * @param[in] newFilterSize		The 2 dimensional size of a single filter. The z dimension is automatically set equal to the neuron dimensions' z value
 * @param[in] newFilterCount	A scalar representing the number of filters in the layer
 * @return						A status code
*/
ConvLayer::ConvLayer(glm::uvec3 const& newNeuronDims, glm::uvec2 const& newFilterSize, uint64_t newFilterCount) {
	this->weightDimensions = glm::uvec3(newFilterSize, newNeuronDims.z);

	// Now continue to setup the layer
	this->setup(newNeuronDims, Type::CONVOLUTION, newFilterCount);
}

/* @brief Perform the feed forward algorithm on this layer using a reference to the next layer. Performs on the GPU with oglopp compute shaders
 * @param[out] nextLayer	A reference to the next layer which will contain the activation result from this layer
 * @return					A reference to this layer
*/
ConvLayer& ConvLayer::feedForward(Layer& nextLayer, oglopp::Compute& compute) {
	compute.use();
	compute.setUIVec3("filterSize", this->weightDimensions);
	compute.setUInt("filterCount", this->weightCountMultiplier);

	Layer::feedForward(nextLayer, compute);
	return *this;
}

/* @brief Perform backpropagation on the layer, given the error/expected value from the next layer.
 * @param[in] nextLayer	A reference to the next layer that will contain either the expected value (if it's OUTPUT), or the carried error from backpropagation (if it's a hidden layer).
 * @param[in] compute	A reference to the compute shader used for backpropagation
 * @return				A reference to this layer object after backpropagation is performed
*/
ConvLayer& ConvLayer::backPropagate(Layer& nextLayer, oglopp::Compute& compute) {
	compute.use();
	compute.setUIVec3("filterSize", this->weightDimensions);
	compute.setUInt("filterCount", this->weightCountMultiplier);

	Layer::backPropagate(nextLayer, compute);
	return *this;
}
