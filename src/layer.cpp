#include "layer.h"
#include "neuron.h"
#include "oglopp/compute.h"
#include "oglopp/ssbo.h"
#include <cmath>
#include <csignal>
#include <iostream>

/* @brief Setup the layer using new neuron dimensions and weight dimensions. Also allow specifying the new layer
 * @param[in] newNeuronDims	The new size of the neurons in 3 dimensional space
 * @param[in] newType		The new type of the layer. Specifies which component of LayerSettings to read
 * @param[in] weightMultiplier	A multiplier to the total weights addressed by weightDims
 * @return					A status code. 0 Upon success, <0 upon failure.
*/
int8_t Layer::setup(glm::uvec3 const& newNeuronDims, Type newType, uint64_t weightMultiplier) {
	if (newNeuronDims.x == 0 || newNeuronDims.y == 0 || newNeuronDims.z == 0) {
		std::cerr << "Failed to setup network layer. Neuron dimensions contained 0 in at least one dimension." << std::endl;
		return -1;
	}


	// Copy over the dimensions of components
	this->weightCountMultiplier = weightMultiplier;
	this->neuronDimensions = newNeuronDims;
	this->type = newType;

	std::cout << "Constructing layer with multiplier " << this->weightCountMultiplier << std::endl;


	// Setup absolute linear sizes for buffer allocation
	const uint64_t NEURON_COUNT = Layer::getTotalElements(newNeuronDims);
	const uint64_t WEIGHT_COUNT = this->weightCountMultiplier * Layer::getTotalElements(this->weightDimensions);

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
	if (WEIGHT_COUNT != 0) {
		float* pWeights = new float[WEIGHT_COUNT];// [weights for neuron 1][weights for neuron 2][weights for neuron 3][[weight 1][weight 2][weight 3] weights for neuron 4]
		if (pWeights == nullptr) {
			std::cerr << "Failed to allocate weights of size " << WEIGHT_COUNT << std::endl;
			return -1;
		}
		std::cout << "Generating weights of size " << WEIGHT_COUNT << std::endl;

		for (uint32_t i=0;i<WEIGHT_COUNT;i++) {
			pWeights[i] = (static_cast<float>(static_cast<double>(rand()) / RAND_MAX) - 0.5) * 2.0;
		}

		this->weights.load(pWeights, sizeof(float) * WEIGHT_COUNT);
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
	compute.setUIVec3("neuronDims", this->neuronDimensions);
	compute.setUIVec3("weightDims", this->weightDimensions);
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
	if (this->weightSize().x * this->weightSize().y * this->weightSize().z != 0) {
		this->getWeights().bind(2);
	}


	compute.use();
	compute.setUIVec3("neuronDims", this->neuronDimensions);
	compute.setUIVec3("weightDims", this->weightDimensions);
	compute.setInt("thisLayerType",	static_cast<int>(this->getType()));
	compute.setInt("thisCount", this->getNeurons().getSize() / sizeof(Neuron));
	compute.setInt("nextLayerType", static_cast<int>(nextLayer.getType()));
	compute.setInt("nextCount", nextLayer.getNeurons().getSize() / sizeof(Neuron));
	compute.setBool("backProp", true);
	compute.setFloat("learningRate", 0.01); //0.003);
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
	// [uint16_t : layer n type]					\/
	// [uint32_t[3] : layer n neuron/bias count]	 |	Layer header
	// [uint32_t[3] : layer n weight count]			 |
	// [ optional layer-specific variables ]		/
	// [float[] : layer n neurons]					\/
	// [float[] : layer n biases]					 |	Layer data
	// [float[] : layer n weights]					/

	// Write the header
	this->writeHeader(stream);

	// Now write the data
	this->writeData(stream);
	return *this;
}

int8_t Layer::writeAdditional(std::fstream& stream) {
	std::cout << "base writeAdditional called" << std::endl;
	return 0;
}

/* @brief Write the layer to
 * @param[in] stream	The stream to write the layer to
 * @return				A reference to this layer object
*/
Layer& Layer::readLayer(std::fstream& stream) {
	// [uint16_t : layer n type]					\/
	// [uint32_t[3] : layer n neuron/bias count]	 |	Layer header
	// [uint32_t[3] : layer n weight count]			 |
	// [ optional layer-specific variables ]		/
	// [float[] : layer n neurons]					\/
	// [float[] : layer n biases]					 |	Layer data
	// [float[] : layer n weights]					/

	// Read the header
	this->readHeader(stream);

	// Read the data
	this->readData(stream);

	return *this;
}

int8_t Layer::readAdditional(std::fstream& stream) {
	return 0;
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
glm::uvec3 const& Layer::neuronSize() {
	return this->neuronDimensions;
}

/* @brief Get the dimensions of the weight list
 * @return A constant reference to the weight dimensions object
*/
glm::uvec3 const& Layer::weightSize() {
	return this->weightDimensions;
}

/* @brief Assig nanother layer to this layer
 * @param[in] copyLayer	The next layer object to copy
 * @return				A reference to this layer object
*/
Layer& Layer::operator=(Layer const& copyLayer) {
	this->copyLayer(copyLayer);

	return *this;
}

/* @brief Assig another layer to this layer
 * @param[in] copyLayer	The next layer object to copy
 * @return				A reference to this layer object
*/
int8_t Layer::copyLayer(Layer const& copyLayer) {
	this->neurons = copyLayer.neurons;
	this->weights = copyLayer.weights;
	this->type = copyLayer.type;
	this->neuronDimensions = copyLayer.neuronDimensions;
	this->weightDimensions = copyLayer.weightDimensions;
	this->weightCountMultiplier = copyLayer.weightCountMultiplier;

	return 0;
}

/* @brief Get the total number of elements from a vec3 dimensions object
 * @return	The total number of elements in a 3 dimensional space
*/
uint64_t Layer::getTotalElements(glm::uvec3 dimensions) {
	return dimensions.x * dimensions.y * dimensions.z;
}

/* @brief Turn a single count into a 3 dimensional list with only a single dimension occupied
 * @param[in] count	The number of elements
 * @return			The count inserted into the x component of a vector
*/
glm::uvec3 Layer::makeSingleDimensional(uint64_t count) {
	return glm::uvec3(count, 1, 1);
}

/* @brief True if this is the last layer (type is OUTPUT. False otherwise)
 * @return	True if .getType() returns Type::OUTPUT.
*/
bool Layer::isLastLayer() const {
	return Type::OUTPUT == this->type;
}

/* @brief Write the layer header to a stream
 * @param[in] stream	The stream to write the layer header to
 * @return				A reference to this layer object
*/
int8_t Layer::writeHeader(std::fstream& stream) {
	// [uint16_t : layer n type]				\/
	// [int[3] : layer n neuron/bias count]		 |	Layer header
	// [int[3] : layer n weight count]			 |
	// [ optional layer-specific variables ]	/

	int8_t res = 0;

	// Write the layer type
	res |= Layer::writeVar(stream, this->getType());

	// Write the size of the neurons
	res |= Layer::writeVar(stream, this->neuronSize().x);
	res |= Layer::writeVar(stream, this->neuronSize().y);
	res |= Layer::writeVar(stream, this->neuronSize().z);
	//std::cout << "Writing neuron dims [" << this->neuronDimensions.x << "] [" << this->neuronDimensions.y << "] [" << this->neuronDimensions.z << "]" << std::endl;

	// Write the size of the weights
	res |= Layer::writeVar(stream, this->weightSize().x);
	res |= Layer::writeVar(stream, this->weightSize().y);
	res |= Layer::writeVar(stream, this->weightSize().z);
	res |= Layer::writeVar(stream, this->weightCountMultiplier);
//	std::cout << "Writing weight dims [" << this->weightDimensions.x << "] [" << this->weightDimensions.y << "] [" << this->weightDimensions.z << "]" << std::endl;
	//std::cout << "write weight multiplier " << this->weightCountMultiplier << std::endl;

	// Each implementation can now write their layer specific variables
	return res;
}

/* @brief Write the layer data to a stream
 * @param[in] stream	The stream to write the layer data to
 * @return				A reference to this layer object
*/
int8_t Layer::writeData(std::fstream& stream) {
	// [float[] : layer n biases]				\	Layer data
	// [float[] : layer n weights]				/

	//std::cout << "Writing " << this->getNeurons().getSize() / sizeof(Neuron) << " biases" << std::endl;

	// Map the neurons
	Neuron* neuronMap = static_cast<Neuron*>(this->neurons.map());
	// Write the biases
	for (size_t i=0;i<this->neurons.getSize() / sizeof(Neuron);i++) {
		Layer::writeVar(stream, neuronMap[i].bias);
	}
	// Unmap
	this->neurons.unmap();

	if (this->weightSize().x * this->weightSize().y * this->weightSize().z > 0) {
		//std::cout << "Writing " << this->getWeights().getSize() / sizeof(float) << " weights" << std::endl;

		// Map the weights
		void* weightMap = this->weights.map();
		// Write the weights
		stream.write(static_cast<char*>(weightMap), this->weights.getSize());
		// Unmmap
		this->weights.unmap();
	}

	return 0;
}

/* @brief Read the layer header from a stream
 * @param[in] stream	The stream to read from
 * @return				A reference to this layer object
*/
int8_t Layer::readHeader(std::fstream& stream) {
	// [uint16_t : layer n type]				\/
	// [int[3] : layer n neuron/bias count]		 |	Layer header
	// [int[3] : layer n weight count]			 |
	// [ optional layer-specific variables ]	/

	int8_t res = 0;

	// Write the layer type
	//res |= Layer::readVar(stream, this->type);

	// Write the size of the neurons
	res |= Layer::readVar(stream, this->neuronDimensions.x);
	res |= Layer::readVar(stream, this->neuronDimensions.y);
	res |= Layer::readVar(stream, this->neuronDimensions.z);
	//std::cout << "Reading neuron dims [" << this->neuronDimensions.x << "] [" << this->neuronDimensions.y << "] [" << this->neuronDimensions.z << "]" << std::endl;

	// Write the size of the weights
	res |= Layer::readVar(stream, this->weightDimensions.x);
	res |= Layer::readVar(stream, this->weightDimensions.y);
	res |= Layer::readVar(stream, this->weightDimensions.z);
	res |= Layer::readVar(stream, this->weightCountMultiplier);
	//std::cout << "Reading weight dims [" << this->weightDimensions.x << "] [" << this->weightDimensions.y << "] [" << this->weightDimensions.z << "]" << std::endl;
	//std::cout << "read weight multiplier " << this->weightCountMultiplier << std::endl;

	// Each implementation can now read thier layer specific variables
	return res;
}

/* @brief Read the layer data from a stream
 * @param[in] stream		The stream to read from
 * @return					A reference to this layer object
*/
int8_t Layer::readData(std::fstream& stream) {
	// [float[] : layer n biases]				\	Layer data
	// [float[] : layer n weights]				/

	const uint32_t BIAS_COUNT = Layer::getTotalElements(this->neuronSize());
	const uint32_t WEIGHT_COUNT = this->weightCountMultiplier * Layer::getTotalElements(this->weightSize());

	//std::cout << "Reading " << BIAS_COUNT << " biases, and " << WEIGHT_COUNT << " weights" << std::endl;

	// Read the neurons
	Neuron* neuronBuf = new Neuron[BIAS_COUNT];
	if (neuronBuf == nullptr) {
		std::cerr << "Failed to allocate neurons of size " << BIAS_COUNT << std::endl;
		return -1;
	}

	// Read biases from the file
	for (uint32_t i=0;i<BIAS_COUNT;i++) {
		neuronBuf[i].value = 0.0;
		neuronBuf[i].expected = 0.0;

		if (Layer::readVar(stream, neuronBuf[i].bias) < 0) {
			std::cerr << "Error reading biases from file? Bad: " << stream.bad() << ", eof: " << stream.eof() << ", fail: " << stream.fail() << " i = " << i << std::endl;
			return -1;
		}
		//std::cout << "i=" << i << " val = " << neuronBuf[i].bias << std::endl;
	}

	// Create a new SSBO based on the buffer
	this->neurons.load(static_cast<void*>(neuronBuf), BIAS_COUNT * sizeof(Neuron));

	// free the neuron/bias allocation
	delete[] neuronBuf;

	if (WEIGHT_COUNT > 0) {
		// Allocate a weight buffer
		float* weightBuf = new float[WEIGHT_COUNT];
		if (weightBuf == nullptr) {
			std::cerr << "Failed to allocate weight buffer of size " << WEIGHT_COUNT << std::endl;
			return -1;
		}

		// Read weight data into buffer
		stream.read(static_cast<char*>(static_cast<void*>(weightBuf)), WEIGHT_COUNT * sizeof(float));

		// Load buffer data into weight ssbo
		this->weights.load(weightBuf, WEIGHT_COUNT * sizeof(float));

		// Free the weight array
		delete[] weightBuf;
	}

	return 0;
}
