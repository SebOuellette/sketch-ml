#include "network.h"
#include "layer.h"
#include "layers/conv_layer.h"
#include "layers/fc_layer.h"
#include "layers/output_layer.h"
#include "layers/pool_layer.h"
#include "netutil.h"
#include "oglopp/compute.h"
#include "oglopp/more_shapes.h"
#include "oglopp/window.h"
#include <cstdlib>
#include <filesystem>
#include <sstream>

Network::Network(size_t inputSize, std::vector<size_t> hiddenSizes, size_t outputSize) {
	//this->setup(inputSize, hiddenSizes, outputSize);
}

Network::Network(std::string const& filename) {
	this->setup(filename);
}

Network::~Network() {
	for(size_t i=0;i<this->monitors.size();i++) {
		delete this->monitors[i];
	}

	while (this->layers.size() > 0) {
		this->popLayer();
	}
}

#define RECTS_NUM_X 3

glm::vec3 calcRectPos(uint32_t index) {
	return glm::vec3(-0.25 * (((index - 1) % RECTS_NUM_X) * 2.1) - 0.27, 0.25 - int((index - 1) / RECTS_NUM_X) * 0.25 * 2.1, 1.0);
}



/* @brief Pop a layer (starting at the back) off of the network
 * @return The number of layers remaining in the list
*/
uint32_t Network::popLayer() {
	// Get a pointer to the
	Layer* rawLayer = this->layers.back();
	if (rawLayer == nullptr) {
		return this->layers.size();
	}

	// Cast the pointer to the proper type, and delete it.
	switch (rawLayer->getType()) {
		case Layer::Type::CONVOLUTION:
			delete[] static_cast<ConvLayer*>(rawLayer);
			break;

		case Layer::Type::FULLY_CONNECTED:
			delete[] static_cast<FCLayer*>(rawLayer);
			break;

		case Layer::Type::OUTPUT:
			delete[] static_cast<OutputLayer*>(rawLayer);
			break;

		case Layer::Type::POOLING:
			delete[] static_cast<PoolLayer*>(rawLayer);
			break;
	}

	// Pop off the back now that we deleted the pointer
	this->layers.pop_back();

	return this->layers.size();
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
	return *this->layers[index];
}

/* @brief Get the number of layers as an unsigned integer. Includes the input and ouput layers
 * @return The number of total layers in the network
*/
size_t Network::size() {
	return this->layers.size();
}

/* @brief Perform a feed forward computation on the network. Performs layer 1, then 2, then 3, etc...
 * @param[in] compute	A reference to a compute shader to use
 * @param[in] fromLayer	The layer to start propagation from. This will act as the 'input' layer. E.x. Can be the middle layer in an autoencoder to just perform decoding.
 * @param[in] toLayer	The layer to finish propagation at. This will act as the 'output' layer. E.x. Can be the middle layer in an autoencoder to just perform encoding.
 * @return	A reference to the output layer storing the calculated result
*/
Layer& Network::feedForward(oglopp::Compute& compute, size_t fromLayer, size_t toLayer) {
	if (this->size() < 2) {
		std::cerr << "Failed to feed forward. The network has fewer than 2 layers." << std::endl;
		return *this->layers[this->size() -1];
	}

	size_t layerStopIndex = std::min(toLayer, this->size() - 2);
	size_t layerStartIndex = std::min(fromLayer, layerStopIndex);

	// We start with the first hidden layer, so start by providing the first layer as the "last" layer
	Layer* nextLayer = nullptr;
	Layer* thisLayer = nullptr;

	// Feed forward each layer one at a time
	for (size_t i=layerStartIndex;i<=layerStopIndex;i++) {
		// Get the current layer
		thisLayer = this->layers[i];

		// Get the next layer
		nextLayer = this->layers[i + 1];

		thisLayer->feedForward(*nextLayer, compute);
	}

	//std::cout << std::endl;

	// Return a reference to the output layer
	return *this->layers[this->size() - 1];
}

/* @brief Perform back propagation on the network
 * @param[in] compute	A reference to a compute shader to use
 * @param[in] fromLayer	The layer to start backpropagation from. This will act as the 'output' layer. E.x. Can be the middle layer in an autoencoder to just backpropagate the encoding phase.
 * @param[in] toLayer	The layer to finish backpropagation at. This will act as the 'input' layer. E.x. Can be the middle layer in an autoencoder to just backpropagate the decoding phase.
 * @return	A reference to the output layer storing the calculated result
*/
Network& Network::backProp(oglopp::Compute& compute, size_t fromLayer, size_t toLayer) {
	if (this->size() < 2) {
		std::cerr << "Failed to backpropagate. The network has fewer than 2 layers." << std::endl;
		return *this;
	}

	// We start with the first hidden layer, so start by providing the first layer as the "last" layer
	Layer* nextLayer = nullptr;
	Layer* thisLayer = nullptr;

	ssize_t layerStartIndex = std::min(fromLayer, this->size()-2);
	ssize_t layerStopIndex = std::min(static_cast<ssize_t>(toLayer), layerStartIndex);

	// Feed forward each layer one at a time
	for (ssize_t i=layerStartIndex;i>=layerStopIndex;i--) {
		// Get the current layer
		nextLayer = this->layers[i + 1];
		thisLayer = this->layers[i];

		// Feed forward the layer given the last layer
		thisLayer->backPropagate(*nextLayer, compute);
	}

	return *this;
}

/* @brief Bind the network to a shader
 * @param[in] shader	The shader object to bind the layers' ssbo objects for display
*/
Network& Network::draw(oglopp::Window& window, oglopp::Shader& shader) {
	// Bind all the layers
	glm::uvec2 layerSize;
	Layer::Type theType = Layer::CONVOLUTION; // Default input type


	for (size_t i=0;i<this->size();i++) {
		this->layers[i]->getNeurons().bind(0);

		if (i < this->monitors.size()) {
			if (i > 0) {
				theType = this->layers[i - 1]->getType();
			}

			switch (theType) {
				case Layer::Type::POOLING: {
					PoolLayer* pLayer = static_cast<PoolLayer*>(this->layers[i-1]);
					layerSize = glm::uvec2(pLayer->neuronSize().x / pLayer->getPoolSize().x, pLayer->neuronSize().y / pLayer->getPoolSize().y);
					break;
				}

				case Layer::Type::CONVOLUTION:
					layerSize = glm::uvec2(this->layers[i]->neuronSize().x, this->layers[i]->neuronSize().y);
					break;

				case Layer::Type::FULLY_CONNECTED:
				case Layer::Type::OUTPUT: {
					double res = ceil(sqrt(this->layers[i]->getNeurons().getSize() / sizeof(Neuron)));

					layerSize = glm::uvec2(res, res);
					break;
				}
			}

			shader.setUInt("layerType", theType);
			shader.setVec2("layerSize", layerSize);
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
std::vector<Layer*>& Network::getLayers() {
	return this->layers;
}

/* @brief Save the network layers to a model file. The model file is tagged using information about the model layers, as well as a timestamp
 * @param[in] directory	The directory to save the file into
 * @return A reference to this network object
*/
Network& Network::save(std::string const& directory) {
	// new
	// [uint32_t : total layer count]				>	Model header
	// [uint16_t : layer n type]					\/					\/
	// [uint32_t[3] : layer n neuron/bias count]	 |					 |	Layer header
	// [uint32_t[3] : layer n weight count]			 | Model Data		 |
	// [ optional layer-specific variables ]		 |					/
	// [float[] : layer n neurons]					 |					\/
	// [float[] : layer n biases]					 |					 |	Layer data
	// [float[] : layer n weights]					/					/

	if (directory.size() > 0) {
		std::filesystem::create_directory(directory);
	}

	// Now get the full filepath
	std::string fullPath = directory + this->networkFilename;
	std::cout << "Saving model to " << fullPath << std::endl;

	// Open the file
	std::fstream file(fullPath, std::ios::out | std::ios::binary);
	if (file.bad()) {
		std::cerr << "Failed to open file!" << std::endl;
		return *this;
	}

	// Write total layer count
	uint32_t totalLayers = this->layers.size(); // includes all layers, including input and ouput
	file.write(static_cast<char*>(static_cast<void*>(&totalLayers)), sizeof(totalLayers));
	std::cout << "Writing [" << totalLayers << "] total layers" << std::endl;

	// Write all layers except input
	for (size_t i=0;i<this->layers.size();i++) {
		writeLayer(file, *this->layers[i]);
	}

	file.close();
	return *this;
}

/* @brief Load network layers from a model file. The file can have any name.
 * @param[in] networkFile	The network file to load
 * @return					A reference to this network object
*/
Network& Network::load(std::string const& networkFile) {
	// new
	// [uint32_t : total layer count]
	// [uint16_t : layer n type]
	// [ optional layer-specific variables ]
	// [uint32_t : layer n neuron/bias count]
	// [uint32_t : layer n weight count]
	// [float[] : layer n neurons]
	// [float[] : layer n biases]
	// [float[] : layer n weights]

	// Now get the full filepath
	std::cout << "Loading model from " << networkFile << std::endl;

	// Open the file
	std::fstream file(networkFile, std::ios::in | std::ios::binary);
	if (file.bad()) {
		return *this;
	}

	// Read the total layer count (Including input and output)
	uint32_t totalLayers = 0;
	file.read(static_cast<char*>(static_cast<void*>(&totalLayers)), sizeof(totalLayers));

	// Read all layers in
	for (size_t i=0;i<totalLayers;i++) {
		//std::cout << "reading type: " << std::flush << readLayer(file, *this) << std::endl;
		readLayer(file, *this);
	}

	file.close();
	return *this;
}

/* @brief Generate a new model path for this network based on the setup layers.
 * @return	A reference to the networkFilename variable after the generated filename has been set
*/
std::string const& Network::generateModelPath() {
	std::cout << "Generating new model path" << std::endl;

	// Generate a filename
	std::ostringstream filename;
	filename << "skml_";
	for (size_t i=0;i<layers.size();i++) {
		filename << this->layers[i]->getNeurons().getSize() / sizeof(Neuron) << "_";
	}
	filename << std::to_string(time(NULL)) << "-" << std::to_string(rand()) << MODEL_EXTENSION;

	this->networkFilename = filename.str();
	return this->networkFilename;
}
