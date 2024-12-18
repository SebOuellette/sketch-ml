#include "netutil.h"

size_t charToIndex(char key) {
	std::cout << "key was " << key << std::endl;
	if (key <= GLFW_KEY_9) {
		return key - GLFW_KEY_0;
	} else {
		return key - GLFW_KEY_A + 10;
	}
}

int saveTrainingElement(oglopp::SSBO& buffer, uint8_t key, std::string const& parentDir) {
	// Generate a filename with time
	std::string dir = parentDir + SAMPLES_DIR;
	std::filesystem::create_directory(dir);

	std::string filename = dir + static_cast<char>(key) + "_" + std::to_string(time(NULL)) + "_" + std::to_string(rand()) + ".raw";
	std::cout << "Saving training element for " << key << " to " << filename << std::endl;

	// Open a file for writing
	std::ofstream file(filename, std::ios::out | std::ios::binary);
	if (file.fail()) {
		std::cerr << "File failed to open" << std::endl;
		return -1;
	}

	// Map the ssbo
	Neuron* neurons = static_cast<Neuron*>(buffer.map());

	// Write to the file
	for (size_t i=0;i<buffer.getSize()/sizeof(Neuron);i++) {
		file.write(static_cast<char*>(static_cast<void*>(&neurons[i].value)), sizeof(neurons[i].value));
	}

	// Ummap ssbo
	buffer.unmap();

	// Close the ifle
	file.close();

	return 0;
}

void loadTrainingFiles(std::vector<std::vector<float>>& files, std::vector<uint32_t>& fileIndices, std::string const& parentDir) {
	std::string dir = parentDir + SAMPLES_DIR;
	std::filesystem::create_directory(dir);

	files.clear();

	// Count files in dir
	uint32_t index = 0;
	std::string filename;
	std::vector<float> fileData;
	float thisItem;

	for (auto const& entry : std::filesystem::directory_iterator(dir)) {
		if (entry.is_regular_file()) {
			filename = entry.path().filename().string();
			std::cout << "Loading training sample " << filename << std::endl;
			fileData.clear();

			// Load the file into a string

			// Open file
			std::ifstream file(dir + filename, std::ios::in | std::ios::binary);
			std::cout << "Opening file [" << index << "] " << dir + filename << std::endl;
			if (file.fail()) {
				std::cerr << "File failed to open" << std::endl;
				return;
			}

			// Read each byte from file
			while (!file.eof() && !file.fail()) {
				file.read(static_cast<char*>(static_cast<void*>(&thisItem)), sizeof(thisItem));

				if (!file.eof() && !file.fail()) {
					fileData.push_back(thisItem);
				}
			}

			// Close file
			file.close();

			// Push the file contents to the vector
			files.push_back(fileData);
			fileIndices.push_back(index++);
		}
	}

	// Shuffle the file indices
	std::cout << "Shuffling training data" << std::endl;
	size_t swapIndex = 0;
	for (size_t i=0;i<fileIndices.size();i++) {
		swapIndex = rand() % fileIndices.size();
		uint32_t temp = fileIndices[i];
		fileIndices[i] = fileIndices[swapIndex];
		fileIndices[swapIndex] = temp;
	}
}

void setExpectedOutput(Network& network) {
	Layer* inputLayer = &network.getLayers().front();
	Layer* outputLayer = &network.getLayers().back();
	Neuron* inputMap = nullptr;
	Neuron* outputMap = nullptr;

	// Map the input buffer
	inputMap = static_cast<Neuron*>(inputLayer->getNeurons().map(oglopp::SSBO::READ));
	// Set the expected values in the final layer
	outputMap = static_cast<Neuron*>(outputLayer->getNeurons().map(oglopp::SSBO::BOTH));

	// Set all the values to 0, unless they match the expected index
	for (size_t l=0;l<outputLayer->getNeurons().getSize() / sizeof(Neuron);l++) {
		outputMap[l].expected = inputMap[l].value;
	}

	// Unmap the neurons
	outputLayer->getNeurons().unmap();
	// Unmap the neurons
	inputLayer->getNeurons().unmap();
}

void doSomeSamples(oglopp::Compute& compute, Network& network, std::string const& parentDir, std::vector<std::vector<float>>& files, std::vector<uint32_t>& fileIndices, size_t& offset, size_t countToDo) {
	std::string dir = parentDir + SAMPLES_DIR;
	std::filesystem::create_directory(dir);

	Layer* inputLayer = &network.getLayers().front();
	Layer* outputLayer = &network.getLayers().back();
	Neuron* inputMap = nullptr;
	Neuron* outputMap = nullptr;

	countToDo = glm::min(countToDo, files.size());

	//size_t expectedIndex = 0;
	for (size_t i=0;i<countToDo;i++) {
		size_t fileIndex = fileIndices[(offset + i) % fileIndices.size()];

		// Map the input buffer
		inputMap = static_cast<Neuron*>(inputLayer->getNeurons().map(oglopp::SSBO::BOTH));

		// Read from the file into the neuron indices
		for (size_t n=0;n<files[fileIndex].size();n++) {
			inputMap[n].value = files[fileIndex][n];
		}

		// Set the expected values in the final layer
		outputMap = static_cast<Neuron*>(outputLayer->getNeurons().map(oglopp::SSBO::BOTH));

		// Set all the values to 0, unless they match the expected index
		for (size_t l=0;l<outputLayer->getNeurons().getSize() / sizeof(Neuron);l++) {
			//neuronMap[l].expected = (l == expectedIndex) ? 1.0 :0.0;
			outputMap[l].expected = inputMap[l].value;
		}

		// Unmap the neurons
		outputLayer->getNeurons().unmap();
		// Unmap the neurons
		inputLayer->getNeurons().unmap();




		// Do forward propagation
		network.feedForward(compute);
		// Now back propagate to train the network
		network.backProp(compute);
	}

	offset = (offset + countToDo) % files.size();
}
