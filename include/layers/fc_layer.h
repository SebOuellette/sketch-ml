#ifndef FC_LAYER_H
#define FC_LAYER_H

#include "layer.h"

// Fully Connected constructor
class FCLayer : public Layer {
public:
	/* @brief Setup a fully-connected layer
	 * @param[in] newNeuronCount	The new size of the neurons in 3 dimensional space
	 * @param[in] newWeightsCount	The number of weights per neuron, also equal to the number of neurons in the next layer
	 * @return						A status code
	*/
	FCLayer(uint64_t const& newNeuronCount, uint64_t newWeightsCount);

	/* @brief Perform the feed forward algorithm on this layer using a reference to the next layer. Performs on the GPU with oglopp compute shaders
	 * @param[out] nextLayer	A reference to the next layer which will contain the activation result from this layer
	 * @return					A reference to this layer
	*/
	FCLayer& feedForward(Layer& nextLayer, oglopp::Compute& compute);

	/* @brief Perform backpropagation on the layer, given the error/expected value from the next layer.
	 * @param[in] nextLayer	A reference to the next layer that will contain either the expected value (if it's OUTPUT), or the carried error from backpropagation (if it's a hidden layer).
	 * @param[in] compute	A reference to the compute shader used for backpropagation
	 * @return				A reference to this layer object after backpropagation is performed
	*/
	FCLayer& backPropagate(Layer& nextLayer, oglopp::Compute& compute);

private:
};

#endif
