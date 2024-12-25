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

	// Default settings struct to be overidden
	struct _LayerSettings {

		/* @brief Pure virtual function to be overloaded
		 * @return	Each implementation of layer settings must return a constant Type, indicating which type this settings block is for. Used for type-safe evaluation at run-time.
	 	*/
		virtual Type getType() const = 0;
	};

	// Get the type. Returns FULLY_CONNECTED
	struct FCSettings : public _LayerSettings {
		// Don't need to know nextLayerNeurons here. We are given that as the weight count in this case.
		uint64_t weightsCount; // The number of weights connecting a single neuron in this layer to all neurons in the next layer

		/* @brief 	Virtual implementation to return FULLY_CONNECTED
		 * @return	Always returns Type::FULLY_CONNECTED
	 	*/
		virtual Type getType() const;
	};

	// Get the type. Returns CONVOLUTION
	struct ConvolutionSettings : public _LayerSettings {
		uint64_t filterCount;	// Number of filters to use. Equal to the number of channels in the next layer/activation map.
		glm::ivec3 filterSize;	// The size of a single filter. The new size of the weights in 3 dimensional space. The size of a single filter for CNN.

		/* @brief 	Virtual implementation to return CONVOLUTION
		 * @return	Always returns Type::CONVOLUTION
	 	*/
		virtual Type getType() const;
	};

	// Settings for a pooling layer
	struct PoolSettings : public _LayerSettings {
		glm::ivec2 size;
		PoolMethod method;

		/* @brief 	Virtual implementation to return POOLING
		 * @return	Always returns Type::POOLING
	 	*/
		virtual Type getType() const;
	};

	typedef union {
		FCSettings settingsFC; // Fully Connected layer - Number of neurons in the next layer
		ConvolutionSettings settingsConv;
		PoolSettings settingsPool;
	} LayerSettings;

	/* @brief Default constructor. By default, the layer is a fully connected input or hidden layer. There's no mathematical difference between input and hidden, but output has no weights.
	 * @param[in] newType	The type of the layer. Used to set up the weights. Sent to the compute shader so it can perform the proper action.
	*/
	Layer() = default;
	~Layer() = default;

	/* @brief Setup the layer using new neuron dimensions and weight dimensions. Also allow specifying the new layer
	 * @param[in] newNeuronDims	The new size of the neurons in 3 dimensional space
	 * @param[in] newType		The new type of the layer. Specifies which component of LayerSettings to read
	 * @param[in] newLayerSettings	A set of variables specific to the variable type.
	 * @return					A status code. 0 Upon success, <0 upon failure.
	*/
	int8_t setup(glm::ivec3 const& newNeuronDims, Type newType, LayerSettings newLayerSettings);

	/* @brief Setup a fully-connected layer
	 * @param[in] newNeuronDims	The new size of the neurons in 3 dimensional space
	 * @param[in] newSettings	The FC settings object contianing extra information
	 * @return					A status code
	*/
	int8_t setupFC(glm::ivec3 const& newNeuronDims, FCSettings newSettings);

	/* @brief Setup a convolutional layer
	 * @param[in] newNeuronDims	The new size of the neurons in 3 dimensional space
	 * @param[in] newSettings	The FC settings object contianing extra information
	 * @return					A status code
	*/
	int8_t setupConv(glm::ivec3 const& newNeuronDims, ConvolutionSettings newSettings);

	/* @brief Setup a pooling layer
	 * @param[in] newNeuronDims	The new size of the input neurons in 3 dimensional space
	 * @param[in] newSettings	The FC settings object contianing extra information
	 * @return					A status code
	*/
	int8_t setupPool(glm::ivec3 const& newNeuronDims, PoolSettings newSettings);

	/* @brief Setup the SSBO with some neurons
	 * @param[in] neuronCount	The number of neurons to randomly initialize and prepare in the SSBO
	 * @param[in] weightCount	The number of weights per neuron (the number of neurons in the last layer)
	*/
	Layer& setup(uint32_t const neuronCount, uint32_t const weightCount);

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
	static constexpr glm::ivec3 makeSingleDimensional(uint64_t count);

	/* @brief True if this is the last layer (type is OUTPUT. False otherwise)
	 * @return	True if .getType() returns Type::OUTPUT.
	*/
	bool isLastLayer() const;

private:
	oglopp::SSBO neurons;
	oglopp::SSBO weights;

	Type type;
	glm::ivec3 neuronDimensions;
	glm::ivec3 weightDimensions;
};

#endif
