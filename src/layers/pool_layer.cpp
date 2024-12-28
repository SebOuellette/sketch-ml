#include "layers/pool_layer.h"

/* @brief The layer to copy. Just copies internal variables (Should also dereference stuff that is not needed anymore)
 * @param[in] copyLayer	A reference to the layer to copy
*/
PoolLayer::PoolLayer(PoolLayer const& copyLayer) {
	this->Layer::operator=(copyLayer);

	this->size = copyLayer.size;
	this->method = copyLayer.method;
}

/* @brief Setup a pooling layer
 * @param[in] newNeuronDims	The new size of the input neurons in 3 dimensional space
 * @param[in] newSettings	The FC settings object contianing extra information
 * @return					A status code
*/
PoolLayer::PoolLayer(glm::ivec3 const& newNeuronDims, glm::ivec2 size, PoolMethod method) {
	// Set the weights to 0
	this->weightDimensions = glm::ivec3(0);

	// Now setup the pooling layer
	this->setup(newNeuronDims, Type::POOLING, 0);
}

/* @brief Perform the feed forward algorithm on this layer using a reference to the next layer. Performs on the GPU with oglopp compute shaders
 * @param[out] nextLayer	A reference to the next layer which will contain the activation result from this layer
 * @return					A reference to this layer
*/
PoolLayer& PoolLayer::feedForward(Layer& nextLayer, oglopp::Compute& compute) {
	compute.use();
	compute.setIVec2("poolSize", this->size);
	compute.setInt("poolMethod", this->method);

	Layer::feedForward(nextLayer, compute);
	return *this;
}

/* @brief Perform backpropagation on the layer, given the error/expected value from the next layer.
 * @param[in] nextLayer	A reference to the next layer that will contain either the expected value (if it's OUTPUT), or the carried error from backpropagation (if it's a hidden layer).
 * @param[in] compute	A reference to the compute shader used for backpropagation
 * @return				A reference to this layer object after backpropagation is performed
*/
PoolLayer& PoolLayer::backPropagate(Layer& nextLayer, oglopp::Compute& compute) {
	compute.use();
	compute.setIVec2("poolSize", this->size);
	compute.setInt("poolMethod", this->method);

	Layer::backPropagate(nextLayer, compute);
	return *this;
}

int8_t PoolLayer::writeAdditional(std::fstream& stream) {
	int8_t res = 0;

	res |= Layer::writeVar(stream, this->size.x);
	res |= Layer::writeVar(stream, this->size.y);

	res |= Layer::writeVar(stream, this->method);
	return res;
}

int8_t PoolLayer::readAdditional(std::fstream& stream) {
	int8_t res = 0;

	res |= Layer::readVar(stream, this->size.x);
	res |= Layer::readVar(stream, this->size.y);

	res |= Layer::readVar(stream, this->method);
	return res;
}
