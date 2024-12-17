#include "network.h"
#include "oglopp/compute.h"
#include "oglopp/more_shapes.h"
#include "oglopp/window.h"
#include <cstdlib>
#include <filesystem>
#include <sstream>

Network::Network(size_t inputSize, std::vector<size_t> hiddenSizes, size_t outputSize) {
	this->setup(inputSize, hiddenSizes, outputSize);
}

Network::Network(std::string const& filename) {
	this->setup(filename);
}

Network::~Network() {
	for(size_t i=0;i<this->monitors.size();i++) {
		delete this->monitors[i];
	}
}

#define RECTS_NUM_X 2

glm::vec3 calcRectPos(uint32_t index) {
	return glm::vec3(-0.25 * (((index - 1) % RECTS_NUM_X) * 2.1) - 0.27, 0.25 - int((index - 1) / RECTS_NUM_X) * 0.25 * 2.1, 1.0);
}

/* @brief Setup the network based on a list of layers and sizes
 * @param[in] inputSize		The input layer size
 * @param[in] layerSizes	The number of neurons in each hidden layer
 * @param[in] outputSize	The ouput layer size
 */
Network& Network::setup(size_t inputSize, std::vector<size_t> hiddenSizes, size_t outputSize) {
	// Generate a filename
	std::ostringstream filename;
	filename << "skml_" << inputSize << "_";
	for (size_t i=0;i<hiddenSizes.size();i++) {
		filename << hiddenSizes[i] << "_";
	}
	filename << outputSize << "_" << std::to_string(time(NULL)) << "-" << std::to_string(rand()) << MODEL_EXTENSION;
	this->networkFilename = filename.str();

	this->layers.resize(2 + hiddenSizes.size());

	// Setup input
	size_t lastSize = inputSize;
	this->layers[0].setup(inputSize, 0);

	// Setup hidden
	for (size_t i=1;i<=hiddenSizes.size();i++) {
		this->layers[i].setup(hiddenSizes[i-1], lastSize);
		lastSize = hiddenSizes[i-1];
	}

	// Setup output
	this->layers[hiddenSizes.size() + 1].setup(outputSize, lastSize);

	return this->setupUI();
}

Network& Network::setup(std::string const& filename) {
	this->networkFilename = filename;

	this->load(this->networkFilename);

	return this->setupUI();
}

Network& Network::setupUI() {
	oglopp::Rectangle* newRect = nullptr;

	// Display input
	newRect = new oglopp::Rectangle;
	newRect->setScale(glm::vec3(1.0, 1.0, 1.0));
	newRect->setPosition(glm::vec3(0.5, 0.0, 1.0));
	this->monitors.push_back(newRect);

	// Display hidden
	for (size_t i=0;i<this->layers.size() - 2;i++) {
		newRect = new oglopp::Rectangle;
		newRect->setScale(glm::vec3(0.5, 0.5, 1.0));
		newRect->setPosition(calcRectPos(i + 1));
		this->monitors.push_back(newRect);
	}

	// Display output
	newRect = new oglopp::Rectangle;
	newRect->setScale(glm::vec3(0.5, 0.5, 1.0));
	newRect->setPosition(calcRectPos(this->layers.size() - 1));
	this->monitors.push_back(newRect);

	return *this;
}

/* @brief True if there was an error with the network, false otherwise
 * @return True if error, false otherwise
*/
bool Network::getError() {
	return this->error;
}

/* @brief Overload the index operator to get a reference to some layer.
 * @param[in] index	The index as an unsigned integer, where index 0 is the input layer, layer 1 is the first hidden layer or the output layer if no hidden layers available
 * @return	A reference to the found layer
*/
Layer& Network::operator[](size_t index) {
	return this->layers[index];
}

/* @brief Get the number of layers as an unsigned integer. Includes the input and ouput layers
 * @return The number of total layers in the network
*/
size_t Network::size() {
	return this->layers.size();
}

/* @brief Perform a feed forward computation on the network. Performs layer 1, then 2, then 3, etc...
 * @param[in] compute	A reference to a compute shader to use
 * @return	A reference to the output layer storing the calculated result
*/
Layer& Network::feedForward(oglopp::Compute& compute) {
	// We start with the first hidden layer, so start by providing the first layer as the "last" layer
	Layer* lastLayer = &this->layers[0];
	Layer* thisLayer = nullptr;

	// Feed forward each layer one at a time
	for (size_t i=1;i<this->size();i++) {
		// Get the current layer
		thisLayer = &this->layers[i];

		// Feed forward the layer given the last layer
		thisLayer->feedForward(*lastLayer, compute);

		// Update last layer to the current layer for the next iteration
		lastLayer = thisLayer;
	}

	// Return a reference to the output layer
	return this->layers[this->size() - 1];
}

/* @brief Perform back propagation on the network
 * @param[in] compute	A reference to a compute shader to use
 * @return	A reference to the output layer storing the calculated result
*/
Network& Network::backProp(oglopp::Compute& compute) {
	// We start with the first hidden layer, so start by providing the first layer as the "last" layer
	Layer* lastLayer = nullptr;
	Layer* thisLayer = nullptr;

	// Feed forward each layer one at a time
	bool isLastLayer = true;
	for (size_t i=this->size()-1;i>0;i--) {
		// Get the current layer
		lastLayer = &this->layers[i-1];
		thisLayer = &this->layers[i];

		// Feed forward the layer given the last layer
		thisLayer->backPropagate(*lastLayer, compute, isLastLayer);
		isLastLayer = false;
	}

	return *this;
}

/* @brief Bind the network to a shader
 * @param[in] shader	The shader object to bind the layers' ssbo objects for display
*/
Network& Network::draw(oglopp::Window& window, oglopp::Shader& shader) {
	// Bind all the layers
	double res = 0;

	for (size_t i=0;i<this->size();i++) {
		this->layers[i].getNeurons().bind(0);

		if (i < this->monitors.size()) {
			res = ceil(sqrt(this->layers[i].getNeurons().getSize() / sizeof(Neuron)));
			//std::cout << "size is " << this->layers[i].getNeurons().getSize() / sizeof(Neuron) << ", res is " << res << std::endl;
			//if (i == this->size() - 1) {
			//	shader.setVec2("layerSize", glm::vec2(this->layers[this->layers.size()-1].getNeurons().getSize() / sizeof(Neuron), 1));
			//} else {
				//shader.setVec2("layerSize", glm::vec2(res, res));
				//}

			shader.setVec2("layerSize", glm::vec2(res, res));

			shader.setVec3("screenPos", this->monitors[i]->getPosition());
			shader.setVec3("screenSize", this->monitors[i]->getScale());
			this->monitors[i]->draw(window, &shader);
		}
	}

	return *this;
}


/* @brief Get a reference to the layers list
 * @return A reference tot he layers list
*/
std::vector<Layer>& Network::getLayers() {
	return this->layers;
}

/* @brief Save the network layers to a model file. The model file is tagged using information about the model layers, as well as a timestamp
 * @param[in] directory	The directory to save the file into
 * @return A reference to this network object
*/
Network& Network::save(std::string const& directory) {
	// [uint32_t : hidden layer count]
	// [uint32_t : input neuron count]
	// [uint32_t : hidden layer 1 neuron count]
	// [uint64_t : input to hidden layer 1 weights count]
	// [float[] : input to hidden layer 1 weights]
	// [float[] : hidden layer 1 biases]
	// [uint32_t : hidden layer N neuron count]
	// [uint64_t : hidden layer N-1 to hidden layer N weights count]
	// [float[] : hidden layer N-1 to hidden layer N weights]
	// [float[] : hidden layer N biases]
	// [uint32_t : output layer neuron count]
	// [uint64_t : hidden layer N to output layer weights count]
	// [float[] : hidden layer N to output layer weights]
	// [float[] : output layer biases]
	//

	if (directory.size() > 0) {
		std::filesystem::create_directory(directory);
	}

	// Now get the full filepath
	std::string fullPath = directory + this->networkFilename;
	std::cout << "Saving model to " << fullPath << std::endl;

	// Open the file
	std::fstream file(fullPath, std::ios::out | std::ios::binary);
	if (file.bad()) {
		std::cerr << "Failed ot open file!" << std::endl;
		return *this;
	}

	// Write hidden layer count
	uint32_t hiddenLayers = this->layers.size() - 2; // includes hidden and output actually but...
	std::cout << "Hidden layers " << hiddenLayers << std::endl;
	file.write(static_cast<char*>(static_cast<void*>(&hiddenLayers)), sizeof(hiddenLayers));

	// Write input neuron count
	uint32_t inputNeuronCount = this->layers[0].getNeurons().getSize() / sizeof(Neuron); // includes hidden and output actually but...
	std::cout << "Input neurons " << inputNeuronCount << std::endl;
	file.write(static_cast<char*>(static_cast<void*>(&inputNeuronCount)), sizeof(inputNeuronCount));

	// Write all layers except input
	for (size_t i=1;i<this->layers.size();i++) {
		std::cout << "Saving layer " << i << std::endl;
		this->layers[i].writeLayer(file);
	}

	file.close();
	return *this;
}

/* @brief Load network layers from a model file. The file can have any name.
 * @param[in] networkFile	The network file to load
 * @return					A reference to this network object
*/
Network& Network::load(std::string const& networkFile) {
	// [uint32_t : hidden layer count]
	// [uint32_t : input neuron count]
	// [uint32_t : hidden layer 1 neuron count]
	// [uint64_t : input to hidden layer 1 weights count]
	// [float[] : input to hidden layer 1 weights]
	// [float[] : hidden layer 1 biases]
	// [uint32_t : hidden layer N neuron count]
	// [uint64_t : hidden layer N-1 to hidden layer N weights count]
	// [float[] : hidden layer N-1 to hidden layer N weights]
	// [float[] : hidden layer N biases]
	// [uint32_t : output layer neuron count]
	// [uint64_t : hidden layer N to output layer weights count]
	// [float[] : hidden layer N to output layer weights]
	// [float[] : output layer biases]
	//

	// Now get the full filepath
	std::cout << "Loading model from " << networkFile << std::endl;

	// Open the file
	std::fstream file(networkFile, std::ios::in | std::ios::binary);
	if (file.bad()) {
		return *this;
	}

	// Write hidden layer count (plus output layer)
	uint32_t hiddenLayers;
	file.read(static_cast<char*>(static_cast<void*>(&hiddenLayers)), sizeof(hiddenLayers));
	std::cout << "Hidden layers " << hiddenLayers << std::endl;

	// Write input neuron count
	uint32_t inputNeuronCount;
	file.read(static_cast<char*>(static_cast<void*>(&inputNeuronCount)), sizeof(inputNeuronCount));
	std::cout << "Input neurons " << inputNeuronCount << std::endl;

	// Setup input layer normally
	this->layers.resize(hiddenLayers + 2);
	this->layers[0].setup(inputNeuronCount, 0);

	// Read all layers except input
	for (size_t i=0;i<=hiddenLayers;i++) {
		std::cout << "Reading " << i + 1 << std::endl;
		this->layers[i + 1].readLayer(file);
	}

	file.close();
	return *this;
}
