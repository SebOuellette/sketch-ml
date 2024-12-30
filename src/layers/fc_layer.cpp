#include "layers/fc_layer.h"


/* @brief The layer to copy. Just copies internal variables (Should also dereference stuff that is not needed anymore)
 * @param[in] copyLayer	A reference to the layer to copy
*/
FCLayer::FCLayer(FCLayer const& copyLayer) {
	this->Layer::operator=(copyLayer);
}

/* @brief Setup a fully-connected layer
 * @param[in] newNeuronCount	The new size of the neurons in 1 dimensional space
 * @param[in] newSettings		The FC settings object contianing extra information
 * @return						A status code
*/
FCLayer::FCLayer(uint64_t const& newNeuronCount, uint64_t weightsCount) {
	// Set the weight dimensions for this layer
	this->weightDimensions = Layer::makeSingleDimensional(weightsCount);
	this->weightCountMultiplier = newNeuronCount;

	// Now continue to setup the layer
	this->setup(Layer::makeSingleDimensional(newNeuronCount), Type::FULLY_CONNECTED, newNeuronCount);
}

/* @brief Perform the feed forward algorithm on this layer using a reference to the next layer. Performs on the GPU with oglopp compute shaders
 * @param[out] nextLayer	A reference to the next layer which will contain the activation result from this layer
 * @return					A reference to this layer
*/
FCLayer& FCLayer::feedForward(Layer& nextLayer, oglopp::Compute& compute) {
	Layer::feedForward(nextLayer, compute);
	return *this;
}

/* @brief Perform backpropagation on the layer, given the error/expected value from the next layer.
 * @param[in] nextLayer	A reference to the next layer that will contain either the expected value (if it's OUTPUT), or the carried error from backpropagation (if it's a hidden layer).
 * @param[in] compute	A reference to the compute shader used for backpropagation
 * @return				A reference to this layer object after backpropagation is performed
*/
FCLayer& FCLayer::backPropagate(Layer& nextLayer, oglopp::Compute& compute) {
	Layer::backPropagate(nextLayer, compute);
	return *this;
}
