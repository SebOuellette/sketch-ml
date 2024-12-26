#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include "layer.h"

// Convolutional Layer
class ConvLayer : public Layer {
public:
	/* @brief Setup a convolutional layer
	 * @param[in] newNeuronDims	The new size of the neurons in 3 dimensional space
	 * @param[in] newFilterSize		The 3 dimensional size of a single filter
	 * @param[in] newFilterCount	A scalar representing the number of filters in the layer
	 * @return						A status code
	*/
	ConvLayer(glm::ivec3 const& newNeuronDims, glm::ivec3 newFilterSize, uint64_t newFilterCount);

	/* @brief Perform the feed forward algorithm on this layer using a reference to the next layer. Performs on the GPU with oglopp compute shaders
	 * @param[out] nextLayer	A reference to the next layer which will contain the activation result from this layer
	 * @return					A reference to this layer
	*/
	ConvLayer& feedForward(Layer& nextLayer, oglopp::Compute& compute);

	/* @brief Perform backpropagation on the layer, given the error/expected value from the next layer.
	 * @param[in] nextLayer	A reference to the next layer that will contain either the expected value (if it's OUTPUT), or the carried error from backpropagation (if it's a hidden layer).
	 * @param[in] compute	A reference to the compute shader used for backpropagation
	 * @return				A reference to this layer object after backpropagation is performed
	*/
	ConvLayer& backPropagate(Layer& nextLayer, oglopp::Compute& compute);

private:
	glm::ivec3 filterSize;
	uint64_t filterCount;
};

#endif
