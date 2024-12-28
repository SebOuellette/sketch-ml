#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include "oglopp/compute.h"
#include "oglopp/more_shapes.h"
#include "oglopp/shader.h"
#include "oglopp/shape.h"
#include "oglopp/window.h"
#include <vector>
#include <oglopp.h>
#include <cstddef>
#include "defines.h"

class Network {
public:
	Network(size_t inputSize, std::vector<size_t> hiddenSizes, size_t outputSize);
	Network(std::string const& filename);
	Network() = default;
	~Network();

	/* @brief Push a new layer onto the network. Also updates the generated model filename
	 * @param[in] layer	A pointer to a layer. This may be a child element, and so it needs to be a pointer
	 * @return 			A reference to this network object
	*/
	template <typename Layer_t>
	int8_t pushLayer(Layer_t const& layer) {
		Layer_t* newLayer = new Layer_t(layer);
		if (newLayer == nullptr) {
			return -1;
		}

		// Push the pointer to the position on the heap
		this->layers.push_back(static_cast<Layer*>(newLayer));

		// Update the model name
		//this->generateModelPath();

		return 0;
	}

	/* @brief Pop a layer (starting at the back) off of the network
	 * @return The number of layers remaining in the list
	*/
	uint32_t popLayer();

	/* @brief Setup the network based on a list of layers and sizes
	 * @param[in] inputSize		The input layer size
	 * @param[in] layerSizes	The number of neurons in each hidden layer
	 * @param[in] outputSize	The ouput layer size
 	 */
	//Network& setup(size_t inputSize, std::vector<size_t> hiddenSizes, size_t outputSize);
	Network& setup(std::string const& filename);
	Network& setupUI();

	/* @brief True if there was an error with the network, false otherwise
	 * @return True if error, false otherwise
	*/
	bool getError();

	/* @brief Overload the index operator to get a reference to some layer.
	 * @param[in] index	The index as an unsigned integer, where index 0 is the input layer, layer 1 is the first hidden layer or the output layer if no hidden layers available
	 * @return	A reference to the found layer
	*/
	Layer& operator[](size_t index);

	/* @brief Get the number of layers as an unsigned integer. Includes the input and ouput layers
	 * @return The number of total layers in the network
	*/
	size_t size();

	/* @brief Perform a feed forward computation on the network. Performs layer 1, then 2, then 3, etc...
	 * @param[in] compute	A reference to a compute shader to use
	 * @param[in] fromLayer	The layer to start propagation from. This will act as the 'input' layer. E.x. Can be the middle layer in an autoencoder to just perform decoding.
	 * @param[in] toLayer	The layer to finish propagation at. This will act as the 'output' layer. E.x. Can be the middle layer in an autoencoder to just perform encoding.
	 * @return	A reference to the output layer storing the calculated result
	*/
	Layer& feedForward(oglopp::Compute& compute, size_t fromLayer = 0, size_t toLayer = static_cast<size_t>(~0x0));


	/* @brief Perform back propagation on the network
	 * @param[in] compute	A reference to a compute shader to use
	 * @param[in] fromLayer	The layer to start backpropagation from. This will act as the 'output' layer. E.x. Can be the middle layer in an autoencoder to just backpropagate the encoding phase.
	 * @param[in] toLayer	The layer to finish backpropagation at. This will act as the 'input' layer. E.x. Can be the middle layer in an autoencoder to just backpropagate the decoding phase.
	 * @return	A reference to the output layer storing the calculated result
	*/
	Network& backProp(oglopp::Compute& compute, size_t fromLayer = static_cast<size_t>(~0x0), size_t toLayer = 0);

	/* @brief Bind the network to a shader
	 * @param[in] shader	The shader object to bind the layers' ssbo objects for display
	*/
	Network& draw(oglopp::Window& window, oglopp::Shader& shader);

	/* @brief Get a reference to the layers list
	 * @return A reference tot he layers list
	*/
	std::vector<Layer*>& getLayers();

	/* @brief Save the network layers to a model file. The model file is tagged using information about the model layers, as well as a timestamp
	 * @param[in] directory	The directory to save the file into
	 * @return A reference to this network object
	*/
	Network& save(std::string const& directory);

	/* @brief Load network layers from a model file. The file can have any name.
	 * @param[in] networkFile	The network file to load
	 * @return					A reference to this network object
	*/
	Network& load(std::string const& networkFile);

	/* @brief Convert a character to an index for classification learning
	 * @param[in] key	The ascii code of the character of the key that was pressed
	 * @return			The index in the output list that corresponds with the pressed key
	*/
	static size_t charToIndex(char key);

	/* @brief Save the layer to a file
	 * @param[in] key		The classification token of the training element. For unsupervised learning this is ignored.
	 * @param[in] parentDir	The directory where the '.raw' sample files are located
	*/
	static int saveTrainingElement(oglopp::SSBO& buffer, uint8_t key, std::string const& parentDir);

	/* @brief Load some data from a raw sample file.
	*/
	static void loadTrainingFiles(std::vector<Layer::Channels>& files, std::vector<uint32_t>& fileIndices, std::string const& parentDir);
	void setExpectedOutput(Network& network);
	void doSomeSamples(oglopp::Compute& compute, Network& network, std::string const& parentDir, std::vector<std::vector<float>>& files, std::vector<uint32_t>& fileIndices, size_t& offset, size_t countToDo);

	/* @brief Generate a new model path for this network based on the setup layers.
	 * @return	A reference to the networkFilename variable after the generated filename has been set
	*/
	std::string const& generateModelPath();


private:
	std::vector<oglopp::Rectangle*> monitors;
	std::vector<Layer*> layers;
	bool error;
	std::string networkFilename;
};

#endif
