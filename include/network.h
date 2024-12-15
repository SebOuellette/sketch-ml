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

	/* @brief Setup the network based on a list of layers and sizes
	 * @param[in] inputSize		The input layer size
	 * @param[in] layerSizes	The number of neurons in each hidden layer
	 * @param[in] outputSize	The ouput layer size
 	 */
	Network& setup(size_t inputSize, std::vector<size_t> hiddenSizes, size_t outputSize);
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
	 * @return	A reference to the output layer storing the calculated result
	*/
	Layer& feedForward(oglopp::Compute& compute);

	/* @brief Perform back propagation on the network
	 * @param[in] compute	A reference to a compute shader to use
	 * @return	A reference to the output layer storing the calculated result
	*/
	Network& backProp(oglopp::Compute& compute);

	/* @brief Bind the network to a shader
	 * @param[in] shader	The shader object to bind the layers' ssbo objects for display
	*/
	Network& draw(oglopp::Window& window, oglopp::Shader& shader);

	/* @brief Get a reference to the layers list
	 * @return A reference tot he layers list
	*/
	std::vector<Layer>& getLayers();

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

private:
	std::vector<oglopp::Rectangle*> monitors;
	std::vector<Layer> layers;
	bool error;
	std::string networkFilename;
};

#endif
