#include "layers/output_layer.h"

/* @brief Setup an output layer
 * @param[in] newNeuronCount	The new size of the input/output neurons in 3 dimensional space
 * @return						A status code
*/
OutputLayer::OutputLayer(uint64_t const& newNeuronCount) {
	// Set the weights to 0
	this->weightDimensions = glm::ivec3(0);

	// Now setup the pooling layer
	this->setup(Layer::makeSingleDimensional(newNeuronCount), Type::OUTPUT, 0);
}
