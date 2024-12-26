#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
#include "oglopp/compute.h"
#include <vector>
#include <cstdlib>
#include <fstream>

class Layer {
public:
	typedef union {
		std::vector<float> bw; // Black & White
		std::vector<float[2]> rg; // Red + Green
		std::vector<float[2]> gb; // Green + blue
		std::vector<float[2]> rb; // Red + blue
		std::vector<float[3]> rgb; // Red + Green + Blue
		std::vector<float[4]> rgba; // Red + Green + Blue + Alpha
	} Channels;

	enum Type {
		FULLY_CONNECTED,	// Fully conected layers are used in ANNs, and in stage 2 of CNNs.
		CONVOLUTION,		// Convolutional layers are used in stage 1 of CNNs
		POOLING,			// Pooling layers are used in stage 1 of CNNs
		OUTPUT				// The output layer of any network. Indicates no weights are allocated.
	};

	enum PoolMethod {
		MAX,	// The maximum value found in the pooled input
		AVG,	// The average value of the pooled input
		MIN		// The minimum value found in the pooled input
	};

	/* @brief Default constructor. By default, the layer is a fully connected input or hidden layer. There's no mathematical difference between input and hidden, but output has no weights.
	 * @param[in] newType	The type of the layer. Used to set up the weights. Sent to the compute shader so it can perform the proper action.
	*/
	Layer() = default;
	~Layer() = default;

	/* @brief Setup the layer using new neuron dimensions and weight dimensions. Also allow specifying the new layer
	 * @param[in] newNeuronDims	The new size of the neurons in 3 dimensional space
	 * @param[in] newType		The new type of the layer. Specifies which component of LayerSettings to read
	 * @param[in] totalWeights	The total number of weights to allocate for this layer. Set to 0 if no weights are required.
	 * @return					A status code. 0 Upon success, <0 upon failure.
	*/
	int8_t setup(glm::ivec3 const& newNeuronDims, Type newType, uint64_t totalWeights);

	/* @brief Perform the feed forward algorithm on this layer using a reference to the next layer. Performs on the GPU with oglopp compute shaders
	 * @param[out] nextLayer	A reference to the next layer which will contain the activation result from this layer
	 * @return					A reference to this layer
	*/
	Layer& feedForward(Layer& nextLayer, oglopp::Compute& compute);

	/* @brief Perform backpropagation on the layer, given the error/expected value from the next layer.
	 * @param[in] nextLayer	A reference to the next layer that will contain either the expected value (if it's OUTPUT), or the carried error from backpropagation (if it's a hidden layer).
	 * @param[in] compute	A reference to the compute shader used for backpropagation
	 * @return				A reference to this layer object after backpropagation is performed
	*/
	Layer& backPropagate(Layer& nextLayer, oglopp::Compute& compute);

	/* @brief Get a reference to the neuron SSBO
	 * @return A reference to the neuron SSBo
	*/
	oglopp::SSBO& getNeurons();

	/* @brief Get a reference to the neuron SSBO
	 * @return A reference to the neuron SSBo
	*/
	oglopp::SSBO& getWeights();

	/* @brief Write the layer to
	 * @param[in] stream	The stream to write the layer to
	 * @return				A reference to this layer object
	*/
	Layer& writeLayer(std::fstream& stream);

	/* @brief Write the layer to
	 * @param[in] stream	The stream to write the layer to
	 * @return				A reference to this layer object
	*/
	Layer& readLayer(std::fstream& stream);

	/* @brief Set the layer type
	 * @param[in] newType	The new type of the layer to set
	 * @return				A reference to this layer
	*/
	Layer& setType(Type const& newType);

	/* @brief Get a constant reference to the type variable
	 * @return	A constant reference to the type variable
	*/
	Type const& getType() const;

	/* @brief Get the dimensions of the neuron list
	 * @return A constant reference to the neuron dimensions object
	*/
	glm::ivec3 const& neuronSize();

	/* @brief Get the dimensions of the weight list
	 * @return A constant reference to the weight dimensions object
	*/
	glm::ivec3 const& weightSize();

	/* @brief Get the total number of elements from a vec3 dimensions object
	 * @return	The total number of elements in a 3 dimensional space
	*/
	static uint64_t getTotalElements(glm::ivec3 dimensions);

	/* @brief Turn a single count into a 3 dimensional list with only a single dimension occupied
	 * @param[in] count	The number of elements
	 * @return			The count inserted into the x component of a vector
	*/
	static glm::ivec3 makeSingleDimensional(uint64_t count);

	/* @brief True if this is the last layer (type is OUTPUT. False otherwise)
	 * @return	True if .getType() returns Type::OUTPUT.
	*/;
	bool isLastLayer() const;

protected:
	oglopp::SSBO neurons;
	oglopp::SSBO weights;

	Type type;
	glm::ivec3 neuronDimensions;
	glm::ivec3 weightDimensions;
};

#endif
