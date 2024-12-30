#include "layers/output_layer.h"

/* @brief The layer to copy. Just copies internal variables (Should also dereference stuff that is not needed anymore)
 * @param[in] copyLayer	A reference to the layer to copy
*/
OutputLayer::OutputLayer(OutputLayer const& copyLayer) {
	this->Layer::operator=(copyLayer);
}

/* @brief Setup an output layer
 * @param[in] newNeuronCount	The new size of the input/output neurons in 3 dimensional space
 * @return						A status code
*/
OutputLayer::OutputLayer(uint64_t const& newNeuronCount) {
	// Set the weights to 0
	this->weightDimensions = glm::ivec3(0);

	// Now setup the pooling layer
	this->setup(Layer::makeSingleDimensional(newNeuronCount), Type::OUTPUT);
}
