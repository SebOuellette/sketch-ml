#include "layer.h"
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
		pNeurons[i].value 	= static_cast<float>(static_cast<double>(rand()) / RAND_MAX);
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

Layer& Layer::backPropagate(Layer& lastLayer, oglopp::Compute& compute) {
	this->getNeurons().bind(0);
	lastLayer.getNeurons().bind(1);
	this->getWeights().bind(2);

	//std::cout << "last count is " << lastLayer.getNeurons().getSize() / sizeof(Neuron) << " while this is " << this->getNeurons().getSize() / sizeof(Neuron) << std::endl;
	compute.use();
	compute.setInt("lastCount", lastLayer.getNeurons().getSize() / sizeof(Neuron));
	compute.setInt("thisCount", this->getNeurons().getSize() / sizeof(Neuron));
	compute.setBool("backProp", true);
	compute.setFloat("learningRate", 0.1);
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
