#ifndef POOL_LAYER_H
#define POOL_LAYER_H

#include "layer.h"

// Pooling Layer
class PoolLayer : public Layer {
public:
	/* @brief Setup a pooling layer
	 * @param[in] newNeuronDims	The new size of the input neurons in 3 dimensional space
	 * @param[in] newSize		The size of the pool reduction. Acts as a divisor on the size of the original input
	 * @param[in] newMethod		The method of pooling. MIN, AVG, or MAX
	 * @return					A status code
	*/
	PoolLayer(glm::ivec3 const& newNeuronDims, glm::ivec2 newSize, PoolMethod newMethod);

	/* @brief Perform the feed forward algorithm on this layer using a reference to the next layer. Performs on the GPU with oglopp compute shaders
	 * @param[out] nextLayer	A reference to the next layer which will contain the activation result from this layer
	 * @return					A reference to this layer
	*/
	PoolLayer& feedForward(Layer& nextLayer, oglopp::Compute& compute);

	/* @brief Perform backpropagation on the layer, given the error/expected value from the next layer.
	 * @param[in] nextLayer	A reference to the next layer that will contain either the expected value (if it's OUTPUT), or the carried error from backpropagation (if it's a hidden layer).
	 * @param[in] compute	A reference to the compute shader used for backpropagation
	 * @return				A reference to this layer object after backpropagation is performed
	*/
	PoolLayer& backPropagate(Layer& nextLayer, oglopp::Compute& compute);

private:
	glm::ivec2 size;
	PoolMethod method;
};

#endif
