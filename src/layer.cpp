#include "layer.h"
#include "neuron.h"
#include "oglopp/compute.h"
#include "oglopp/ssbo.h"
#include <iostream>

/* @brief Setup the SSBO with some neurons
 * @param[in] neuronCount	The number of neurons to randomly initialize and prepare in the SSBO
 * @param[in] weightCount	The number of weights per neuron (the number of neurons in the last layer)
*/
Layer& Layer::setup(uint32_t const neuronCount, uint32_t const weightCount) {
	if (neuronCount == 0) {
		return *this;
	}

	// Allocate some neurons
	Neuron* pNeurons = new Neuron[neuronCount];

	for (uint32_t i=0;i<neuronCount;i++) {
		pNeurons[i].bias 	= static_cast<float>(static_cast<double>(rand()) / RAND_MAX);
		//neurons[i].weight 	= static_cast<float>(static_cast<double>(rand()) / RAND_MAX);
		pNeurons[i].value 	= 0.0; //static_cast<float>(static_cast<double>(rand()) / RAND_MAX);
		pNeurons[i].expected = 0.0; //= (static_cast<float>(static_cast<double>(rand()) / RAND_MAX) - 0.5) * 2.0;
	}

	this->neurons.load(pNeurons, sizeof(Neuron) * neuronCount);
	delete[] pNeurons;

	// Allocate the weights
	if (weightCount == 0) {
		return *this;
	}

	const size_t NUM_WEIGHTS = neuronCount * weightCount;
	float* pWeights = new float[NUM_WEIGHTS];// [weights for neuron 1][weights for neuron 2][weights for neuron 3][[weight 1][weight 2][weight 3] weights for neuron 4]

	for (uint32_t i=0;i<NUM_WEIGHTS;i++) {
		pWeights[i] = (static_cast<float>(static_cast<double>(rand()) / RAND_MAX) - 0.5) * 2.0;
	}

	this->weights.load(pWeights, sizeof(float) * NUM_WEIGHTS);
	delete[] pWeights;
	return *this;
}

/* @brief Setup the layer using an SSBO
 * @param[in] neuronCopy	A constant reference to an SSBO object to copy into the neurons
 * @return					A reference to this layer object
*/
Layer& Layer::setup(oglopp::SSBO const& neuronCopy) {
	return *this;
}

/* @brief Perform the feed forward algorithm on this layer using a reference to the previous layer. Performs on the GPU with oglopp compute shaders
 * @param[in] lastLayer	A reference to the last layer to be fed into this layer
 * @return				A reference to this layer
*/
Layer& Layer::feedForward(Layer& lastLayer, oglopp::Compute& compute) {
	this->getNeurons().bind(0);
	lastLayer.getNeurons().bind(1);
	this->getWeights().bind(2);

	//std::cout << "last count is " << lastLayer.getNeurons().getSize() / sizeof(Neuron) << " while this is " << this->getNeurons().getSize() / sizeof(Neuron) << std::endl;
	compute.use();
	compute.setInt("lastCount", lastLayer.getNeurons().getSize() / sizeof(Neuron));
	compute.setInt("thisCount", this->getNeurons().getSize() / sizeof(Neuron));
	compute.setBool("backProp", false);
	compute.dispatch(this->neurons.getSize() / sizeof(Neuron), 1);

	oglopp::SSBO::unbind();

	return *this;
}

Layer& Layer::backPropagate(Layer& lastLayer, oglopp::Compute& compute, bool isLastLayer) {
	this->getNeurons().bind(0);
	lastLayer.getNeurons().bind(1);
	this->getWeights().bind(2);

	//std::cout << "last count is " << lastLayer.getNeurons().getSize() / sizeof(Neuron) << " while this is " << this->getNeurons().getSize() / sizeof(Neuron) << std::endl;
	compute.use();
	compute.setBool("isLastLayer", isLastLayer);
	compute.setInt("lastCount", lastLayer.getNeurons().getSize() / sizeof(Neuron));
	compute.setInt("thisCount", this->getNeurons().getSize() / sizeof(Neuron));
	compute.setBool("backProp", true);
	compute.setFloat("learningRate", 0.003);
	compute.dispatch(lastLayer.getNeurons().getSize() / sizeof(Neuron), 1);

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
	std::cout << "Allocating weights " << weightsSize << std::endl;
	float* weights = new float[weightsSize];
	if (weights == nullptr) {
		std::cerr << "Failed to allocate weights buffer during read of file" << std::endl;
		return *this;
	}
	stream.read(static_cast<char*>(static_cast<void*>(weights)), weightsSize * sizeof(float));
	this->weights.load(weights, weightsSize * sizeof(float));
	delete[] weights;

	// Write the biases
	std::cout << "Allocating neurons " << neuronSize << std::endl;
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
