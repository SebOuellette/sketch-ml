#include "layer.h"
#include "neuron.h"
#include "oglopp/compute.h"
#include "oglopp/ssbo.h"
#include <csignal>
#include <iostream>

/* @brief Setup the layer using new neuron dimensions and weight dimensions. Also allow specifying the new layer
 * @param[in] newNeuronDims	The new size of the neurons in 3 dimensional space
 * @param[in] newType		The new type of the layer. Specifies which component of LayerSettings to read
 * @param[in] totalWeights	The total number of weights to allocate for this layer. Set to 0 if no weights are required.
 * @return					A status code. 0 Upon success, <0 upon failure.
*/
int8_t Layer::setup(glm::ivec3 const& newNeuronDims, Type newType, uint64_t totalWeights) {
	if (newNeuronDims.x == 0 || newNeuronDims.y == 0 || newNeuronDims.z == 0) {
		std::cerr << "Failed to setup network layer. Neuron dimensions contained 0 in at least one dimension." << std::endl;
		return -1;
	}

	// Copy over the dimensions of components
	this->neuronDimensions = newNeuronDims;
	this->type = newType;

	// Setup absolute linear sizes for buffer allocation
	const uint64_t NEURON_COUNT = Layer::getTotalElements(newNeuronDims);

	// Allocate some neurons
	Neuron* pNeurons = new Neuron[NEURON_COUNT];

	// Initialize the data
	for (uint32_t i=0;i<NEURON_COUNT;i++) {
		pNeurons[i].bias 	= static_cast<float>(static_cast<double>(rand()) / RAND_MAX);
		pNeurons[i].value 	= 0.0;
		pNeurons[i].expected = 0.0;
	}

	// Load the neurons into the SSBO
	this->neurons.load(pNeurons, sizeof(Neuron) * NEURON_COUNT);
	delete[] pNeurons;

	// Only allocate weights if we decided that this layer should have weights.
	if (totalWeights != 0) {
		float* pWeights = new float[totalWeights];// [weights for neuron 1][weights for neuron 2][weights for neuron 3][[weight 1][weight 2][weight 3] weights for neuron 4]

		for (uint32_t i=0;i<totalWeights;i++) {
			pWeights[i] = (static_cast<float>(static_cast<double>(rand()) / RAND_MAX) - 0.5) * 2.0;
		}

		this->weights.load(pWeights, sizeof(float) * totalWeights);
		delete[] pWeights;
	}

	return 0;
}

/* @brief Perform the feed forward algorithm on this layer using a reference to the next layer. Performs on the GPU with oglopp compute shaders
 * @param[out] nextLayer	A reference to the next layer which will contain the activation result from this layer
 * @return					A reference to this layer
*/
Layer& Layer::feedForward(Layer& nextLayer, oglopp::Compute& compute) {
	this->getNeurons().bind(0);
	nextLayer.getNeurons().bind(1);
	this->getWeights().bind(2);

	//std::cout << "last count is " << lastLayer.getNeurons().getSize() / sizeof(Neuron) << " while this is " << this->getNeurons().getSize() / sizeof(Neuron) << std::endl;
	compute.use();
	compute.setIVec3("neuronDims", this->neuronDimensions);
	compute.setIVec3("weightDims", this->weightDimensions);
	compute.setInt("nextCount", nextLayer.getNeurons().getSize() / sizeof(Neuron));
	compute.setInt("thisCount", this->getNeurons().getSize() / sizeof(Neuron));
	compute.setBool("backProp", false);
	compute.setInt("thisLayerType", this->getType());
	//compute.dispatch(nextLayer.getNeurons().getSize() / sizeof(Neuron), 1);
	compute.dispatch(nextLayer.neuronSize());

	oglopp::SSBO::unbind();

	return *this;
}

/* @brief Perform backpropagation on the layer, given the error/expected value from the next layer.
 * @param[in] nextLayer	A reference to the next layer that will contain either the expected value (if it's OUTPUT), or the carried error from backpropagation (if it's a hidden layer).
 * @param[in] compute	A reference to the compute shader used for backpropagation
 * @return				A reference to this layer object after backpropagation is performed
*/
Layer& Layer::backPropagate(Layer& nextLayer, oglopp::Compute& compute) {
	this->getNeurons().bind(0);
	nextLayer.getNeurons().bind(1);
	this->getWeights().bind(2);

	compute.use();
	compute.setIVec3("neuronDims", this->neuronDimensions);
	compute.setIVec3("weightDims", this->weightDimensions);
	compute.setInt("thisLayerType",	static_cast<int>(this->getType()));
	compute.setInt("thisCount", this->getNeurons().getSize() / sizeof(Neuron));
	compute.setInt("nextLayerType", static_cast<int>(nextLayer.getType()));
	compute.setInt("nextCount", nextLayer.getNeurons().getSize() / sizeof(Neuron));
	compute.setBool("backProp", true);
	compute.setFloat("learningRate", 0.003);
	//compute.dispatch(this->getNeurons().getSize() / sizeof(Neuron), 1);
	compute.dispatch(this->neuronSize());

	oglopp::SSBO::unbind();

	return *this;
}

/* @brief Get a reference to the neuron SSBO
 * @return A reference to the neuron SSBo
*/
oglopp::SSBO& Layer::getNeurons() {
	return this->neurons;
}

/* @brief Get a reference to the neuron SSBO
 * @return A reference to the neuron SSBo
*/
oglopp::SSBO& Layer::getWeights() {
	return this->weights;
}

/* @brief Write the layer to
 * @param[in] stream	The stream to write the layer to
 * @return				A reference to this layer object
*/
Layer& Layer::writeLayer(std::fstream& stream) {
	// [uint32_t : Layer neuron count]
	// [uint64_t : Last layer to this layer weights count]
	// [float[] : Last layer to this layer weights]
	// [float[] : Layer biases]

	// Write the size of the neurons
	uint32_t neuronSize = this->neurons.getSize() / sizeof(Neuron);
	stream.write(static_cast<char*>(static_cast<void*>(&neuronSize)), sizeof(neuronSize));

	// Write the size of the weights
	uint64_t weightsSize = this->weights.getSize() / sizeof(float);
	stream.write(static_cast<char*>(static_cast<void*>(&weightsSize)), sizeof(weightsSize));

	// Write the weights
	void* weightsMap = this->weights.map();
	stream.write(static_cast<char*>(weightsMap), this->weights.getSize());
	this->weights.unmap();

	// Write the biases
	Neuron* neuronsMap = static_cast<Neuron*>(this->neurons.map());
	for (size_t i=0;i<neuronSize;i++) {
		// Write each bias
		stream.write(static_cast<char*>(static_cast<void*>(&neuronsMap[i].bias)), sizeof(float));
	}
	this->neurons.unmap();


	return *this;
}

/* @brief Write the layer to
 * @param[in] stream	The stream to write the layer to
 * @return				A reference to this layer object
*/
Layer& Layer::readLayer(std::fstream& stream) {
	// [uint32_t : Layer neuron count]
	// [uint64_t : Last layer to this layer weights count]
	// [float[] : Last layer to this layer weights]
	// [float[] : Layer biases]

	// Write the size of the neurons
	uint32_t neuronSize = 0;
	stream.read(static_cast<char*>(static_cast<void*>(&neuronSize)), sizeof(neuronSize));

	// Write the size of the weights
	uint64_t weightsSize = 0;
	stream.read(static_cast<char*>(static_cast<void*>(&weightsSize)), sizeof(weightsSize));

	// Read the weights
	///std::cout << "Allocating weights " << weightsSize << std::endl;
	float* weights = new float[weightsSize];
	if (weights == nullptr) {
		std::cerr << "Failed to allocate weights buffer during read of file" << std::endl;
		return *this;
	}
	stream.read(static_cast<char*>(static_cast<void*>(weights)), weightsSize * sizeof(float));
	this->weights.load(weights, weightsSize * sizeof(float));
	delete[] weights;

	// Write the biases
	//std::cout << "Allocating neurons " << neuronSize << std::endl;
	Neuron* neurons = new Neuron[neuronSize];
	if (neurons == nullptr) {
		std::cerr << "Failed to allocate weights buffer during read of file" << std::endl;
		return *this;
	}
	for (size_t i=0;i<neuronSize;i++) {
		// Write each bias
		stream.read(static_cast<char*>(static_cast<void*>(&neurons[i].bias)), sizeof(float));
		neurons[i].expected = 0.0; // Just initialize the data to something
		neurons[i].value = 0.0;
	}
	this->neurons.load(neurons, neuronSize * sizeof(Neuron));
	delete[] neurons;
	return *this;
}

/* @brief Set the layer type. Does not setup or destroy weights.
 * @param[in] newType	The new type of the layer to set
 * @return				A reference to this layer
*/
Layer& Layer::setType(Type const& newType) {
	this->type = newType;

	return *this;
}

/* @brief Get a constant reference to the type variable
 * @return	A constant reference to the type variable
*/
Layer::Type const& Layer::getType() const {
	return this->type;
}

/* @brief Get the dimensions of the neuron list
 * @return A constant reference to the neuron dimensions object
*/
glm::ivec3 const& Layer::neuronSize() {
	return this->neuronDimensions;
}

/* @brief Get the dimensions of the weight list
 * @return A constant reference to the weight dimensions object
*/
glm::ivec3 const& Layer::weightSize() {
	return this->weightDimensions;
}

/* @brief Get the total number of elements from a vec3 dimensions object
 * @return	The total number of elements in a 3 dimensional space
*/
uint64_t Layer::getTotalElements(glm::ivec3 dimensions) {
	return dimensions.x * dimensions.y * dimensions.z;
}

/* @brief Turn a single count into a 3 dimensional list with only a single dimension occupied
 * @param[in] count	The number of elements
 * @return			The count inserted into the x component of a vector
*/
glm::ivec3 Layer::makeSingleDimensional(uint64_t count) {
	return glm::ivec3(count, 1, 1);
}

/* @brief True if this is the last layer (type is OUTPUT. False otherwise)
 * @return	True if .getType() returns Type::OUTPUT.
*/
bool Layer::isLastLayer() const {
	return Type::OUTPUT == this->type;
}
