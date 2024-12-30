#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"
#include "oglopp/compute.h"
#include <vector>
#include <cstdlib>
#include <fstream>
#include <cstdint>

#include <iostream>

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

	enum Type : uint32_t {
		FULLY_CONNECTED = 0x00,	// Fully conected layers are used in ANNs, and in stage 2 of CNNs.
		CONVOLUTION,		// Convolutional layers are used in stage 1 of CNNs
		POOLING,			// Pooling layers are used in stage 1 of CNNs
		OUTPUT				// The output layer of any network. Indicates no weights are allocated.
	};

	enum PoolMethod : uint32_t {
		MAX = 0x00,	// The maximum value found in the pooled input
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
	 * @param[in] weightMultiplier	A multiplier to the total weights addressed by weightDims
	 * @return					A status code. 0 Upon success, <0 upon failure.
	*/
	int8_t setup(glm::uvec3 const& newNeuronDims, Type newType, uint64_t weightMultiplier = 1);

	/* @brief Perform the feed forward algorithm on this layer using a reference to the next layer. Performs on the GPU with oglopp compute shaders
	 * @param[out] nextLayer	A reference to the next layer which will contain the activation result from this layer
	 * @return					A reference to this layer
	*/
	virtual Layer& feedForward(Layer& nextLayer, oglopp::Compute& compute);

	/* @brief Perform backpropagation on the layer, given the error/expected value from the next layer.
	 * @param[in] nextLayer	A reference to the next layer that will contain either the expected value (if it's OUTPUT), or the carried error from backpropagation (if it's a hidden layer).
	 * @param[in] compute	A reference to the compute shader used for backpropagation
	 * @return				A reference to this layer object after backpropagation is performed
	*/
	virtual  Layer& backPropagate(Layer& nextLayer, oglopp::Compute& compute);

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
	virtual Layer& writeLayer(std::fstream& stream);
	virtual int8_t writeAdditional(std::fstream& stream);

	/* @brief Write the layer to
	 * @param[in] stream	The stream to write the layer to
	 * @return				A reference to this layer object
	*/
	virtual Layer& readLayer(std::fstream& stream);
	virtual int8_t readAdditional(std::fstream& stream);

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
	glm::uvec3 const& neuronSize();

	/* @brief Get the dimensions of the weight list
	 * @return A constant reference to the weight dimensions object
	*/
	glm::uvec3 const& weightSize();

	/* @brief Assig nanother layer to this layer
	 * @param[in] copyLayer	The next layer object to copy
	 * @return				A reference to this layer object
	*/
	Layer& operator=(Layer const& copyLayer);

	/* @brief Assig another layer to this layer
	 * @param[in] copyLayer	The next layer object to copy
	 * @return				A reference to this layer object
	*/
	int8_t copyLayer(Layer const& copyLayer);

	/* @brief Get the total number of elements from a vec3 dimensions object
	 * @return	The total number of elements in a 3 dimensional space
	*/
	static uint64_t getTotalElements(glm::uvec3 dimensions);

	/* @brief Turn a single count into a 3 dimensional list with only a single dimension occupied
	 * @param[in] count	The number of elements
	 * @return			The count inserted into the x component of a vector
	*/
	static glm::uvec3 makeSingleDimensional(uint64_t count);

	/* @brief True if this is the last layer (type is OUTPUT. False otherwise)
	 * @return	True if .getType() returns Type::OUTPUT.
	*/;
	bool isLastLayer() const;

	/* @brief Read some data from a stream into a variable (CPU Endianness /shrug)
	 * @param[out] 	output	A reference to the output variable that the data will be read into
	 * @param[in]	stream	A reference to the fstream to read from
	 * @return				-1 if the stream was bad after reading, 0 otherwise
	*/
	template <typename T>
	inline static int8_t readVar(std::fstream& stream, T& output) {
		stream.read(static_cast<char*>(static_cast<void*>(&output)), sizeof(output));
		//std::cout << "Read var '" << output << "'" << std::endl;

		return (stream.bad() || stream.eof()) ? -1 : 0;
	}

	/* @brief Write some data to a stream from a variable (CPU Endianness /shrug)
	 * @param[out] 	output	A reference to the output variable that the data will be read from
	 * @param[in]	stream	A reference to the fstream to write to
	 * @return				-1 if the stream was bad after reading, 0 otherwise
	*/
	template <typename T>
	inline static int8_t writeVar(std::fstream& stream, T const& output) {
		stream.write(static_cast<const char*>(static_cast<const void*>(&output)), sizeof(output));

		return (stream.bad()) ? -1 : 0;
	}

	/* @brief Write the layer header to a stream
	 * @param[in] stream	The stream to write the layer header to
	 * @return				A reference to this layer object
	*/
	int8_t writeHeader(std::fstream& stream);

	/* @brief Write the layer data to a stream
	 * @param[in] stream	The stream to write the layer data to
	 * @return				A reference to this layer object
	*/
	int8_t writeData(std::fstream& stream);

	/* @brief Read the layer header from a stream
	 * @param[in] stream	The stream to read from
	 * @return				A reference to this layer object
	*/
	int8_t readHeader(std::fstream& stream);

	/* @brief Read the layer data from a stream
	 * @param[in] stream		The stream to read from
	 * @param[in] biasCount		The number of neuron biases to read from the file
	 * @param[in] weightCount	The number of weights to read from the file
	 * @return					A reference to this layer object
	*/
	int8_t readData(std::fstream& stream);

protected:

	oglopp::SSBO neurons;
	oglopp::SSBO weights;

	Type type;
	glm::uvec3 neuronDimensions;
	glm::uvec3 weightDimensions;
	uint weightCountMultiplier;
};

#endif
