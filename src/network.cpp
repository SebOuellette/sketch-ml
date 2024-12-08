#include "network.h"
#include "oglopp/compute.h"
#include "oglopp/more_shapes.h"
#include "oglopp/window.h"
#include <cstdlib>

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

/* @brief Setup the network based on a list of layers and sizes
 * @param[in] inputSize		The input layer size
 * @param[in] layerSizes	The number of neurons in each hidden layer
 * @param[in] outputSize	The ouput layer size
 */
Network& Network::setup(size_t inputSize, std::vector<size_t> hiddenSizes, size_t outputSize) {
	this->layers.resize(2 + hiddenSizes.size());
	oglopp::Rectangle* newRect = nullptr;

	// Setup input
	size_t lastSize = inputSize;
	this->layers[0].setup(inputSize, 0);

	newRect = new oglopp::Rectangle;
	newRect->setScale(glm::vec3(1.0, 1.0, 1.0));
	newRect->setPosition(glm::vec3(1.0, 0.0, 1.0));
	this->monitors.push_back(newRect);

	// Setup hidden
	for (size_t i=1;i<=hiddenSizes.size();i++) {
		this->layers[i].setup(hiddenSizes[i-1], lastSize);
		lastSize = hiddenSizes[i-1];

		newRect = new oglopp::Rectangle;
		newRect->setScale(glm::vec3(0.5, 0.5, 1.0));
		newRect->setPosition(glm::vec3(-0.25 * (((i - 1) % 3) * 2.1) + 0.20, 0.25 - int((i - 1) / 3) * 0.25 * 2.1, 1.0));
		this->monitors.push_back(newRect);
	}

	// Setup output
	this->layers[hiddenSizes.size() + 1].setup(outputSize, lastSize);

	newRect = new oglopp::Rectangle;
	newRect->setScale(glm::vec3(1.0, 0.1, 1.0));
	newRect->setPosition(glm::vec3(1.0, -0.6, 1.0));
	this->monitors.push_back(newRect);

	return *this;
}

Network& Network::setup(std::string const& filename) {
	std::cerr << "Filename network loading is not implemented" << std::endl;
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
	for (size_t i=this->size()-1;i>0;i--) {
		// Get the current layer
		lastLayer = &this->layers[i-1];
		thisLayer = &this->layers[i];

		// Feed forward the layer given the last layer
		thisLayer->backPropagate(*lastLayer, compute);
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
			if (i == this->size() - 1) {
				shader.setVec2("layerSize", glm::vec2(this->layers[this->layers.size()-1].getNeurons().getSize() / sizeof(Neuron), 1));
			} else {
				shader.setVec2("layerSize", glm::vec2(res, res));
			}

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
