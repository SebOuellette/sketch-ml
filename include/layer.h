#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
#include "oglopp/compute.h"
#include <vector>
#include <cstdlib>

class Layer {
public:
	Layer() = default;
	~Layer() = default;

	/* @brief Setup the SSBO with some neurons
	 * @param[in] neuronCount	The number of neurons to randomly initialize and prepare in the SSBO
	 * @param[in] weightCount	The number of weights per neuron (the number of neurons in the last layer)
	*/
	Layer& setup(uint32_t const neuronCount, uint32_t const weightCount);

	/* @brief Setup the layer using an SSBO
	 * @param[in] neuronCopy	A constant reference to an SSBO object to copy into the neurons
	 * @return					A reference to this layer object
	*/
	Layer& setup(oglopp::SSBO const& neuronCopy);

	/* @brief Perform the feed forward algorithm on this layer using a reference to the previous layer. Performs on the GPU with oglopp compute shaders
	 * @param[in] lastLayer	A reference to the last layer to be fed into this layer
	 * @return				A reference to this layer
	*/
	Layer& feedForward(Layer& lastLayer, oglopp::Compute& compute);


	Layer& backPropagate(Layer& lastLayer, oglopp::Compute& compute, bool isLastLayer);

	/* @brief Get a reference to the neuron SSBO
	 * @return A reference to the neuron SSBo
	*/
	oglopp::SSBO& getNeurons();

	/* @brief Get a reference to the neuron SSBO
	 * @return A reference to the neuron SSBo
	*/
	oglopp::SSBO& getWeights();

private:
	oglopp::SSBO neurons;
	oglopp::SSBO weights;
};

#endif
