#include <chrono>
#include <cstddef>
#include <glm/common.hpp>
#include <oglopp.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <thread>

#include "network.h"
#include "neuron.h"
#include "oglopp/camera.h"
#include "oglopp/compute.h"
#include "oglopp/more_shapes.h"
#include "oglopp/ssbo.h"

#include "stb_image.h"

using namespace oglopp;

#define RESOLUTION	(32) // RESOLUTION x RESOLUTION pixels
#define PIXELS		(RESOLUTION * RESOLUTION)
#define SAMPLES_DIR	"samples/"

#define TRAINSIZE 5

class InputBuffer {
public:
	static Window* windowPtr;
	static float drawSize;

	static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
		drawSize += yoffset * 3;
		drawSize = glm::max(drawSize, 1.f);
	}
};

Window* InputBuffer::windowPtr  = nullptr;
float InputBuffer::drawSize = 30;

size_t charToIndex(char key) {
	std::cout << "key was " << key << std::endl;
	if (key <= GLFW_KEY_9) {
		return key - GLFW_KEY_0;
	} else {
		return key - GLFW_KEY_A + 10;
	}
}

int saveTrainingElement(SSBO& buffer, uint8_t key, std::string const& parentDir) {
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
				fileData.push_back(thisItem);
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

void doSomeSamples(Compute& compute, Network& network, std::string const& parentDir, std::vector<std::vector<float>>& files, std::vector<uint32_t>& fileIndices, size_t& offset, size_t countToDo) {
	std::string dir = parentDir + SAMPLES_DIR;
	std::filesystem::create_directory(dir);

	Layer* inputLayer = &network.getLayers().front();
	Layer* outputLayer = &network.getLayers().back();
	Neuron* inputMap = nullptr;
	Neuron* outputMap = nullptr;

	countToDo = glm::min(countToDo, files.size());

	size_t expectedIndex = 0;
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
		inputLayer->getNeurons().unmap();

		// Unmap the neurons
		outputLayer->getNeurons().unmap();


		// Do forward propagation
		network.feedForward(compute);
		// Now back propagate to train the network
		network.backProp(compute);
	}

	offset = (offset + countToDo) % files.size();
}

int main(int argc, char** argv) {
	srand(time(NULL));

	// Setup some window options to make it invisible
	Window::Settings options;
	options.visible = true;
	options.doFaceCulling = false;
	options.modifyPointSize = true;
	options.clearColor = glm::vec4(glm::vec3(0.05), 1.0);

	// Create the window
	Window window;
	window.create(800, 800, "Handwritten Digit Recognition", options);
	InputBuffer::windowPtr = &window;
	glfwSetScrollCallback(window.getWindow(), InputBuffer::scrollCallback);

	std::string MY_PATH = std::filesystem::canonical("/proc/self/exe");
	std::size_t pos = MY_PATH.find_last_of('/');
	if (pos == std::string::npos) {
		std::cerr << "Current path does not have a '/' character.. I don't know how to handle this. Path was: '" << MY_PATH << "'" << std::endl;
		return 1;
	}
	MY_PATH = MY_PATH.substr(0, pos + 1);

	// Initialize our shader object(s)
	Compute compute((MY_PATH + "shaders/compute.glsl").c_str(), ShaderType::FILE);
	Shader shader((MY_PATH + "shaders/vertex.glsl").c_str(), (MY_PATH + "shaders/fragment.glsl").c_str(), ShaderType::FILE);

	std::this_thread::sleep_for(std::chrono::duration(std::chrono::seconds(1)));

	// Create a list of vertices with just an ID. the position will be provided in an SSBO calculated by a compute shader
	Rectangle screen;
	screen.setScale(glm::vec3(1.0, 1.0, 1.0));
	screen.setPosition(glm::vec3(0.0, 0.0, 1.0));

	// Create a network
	//Network network(32*32, {100*100, 50*50, 20*20}, 32*32);
	Network network; //(32*32, {50*50, 20*20, 10, 20*20, 50*50}, 32*32);
	if (argc < 2) {
		network.setup(32*32, {50*50, 20*20, 10, 20*20, 50*50}, 32*32);
	} else {
		network.setup(argv[1]);
	}

	int width, height;
	int8_t keyDown = 0;
	bool justPressed = false;
	bool enterPressed = false;
	bool trainingToggle = false;
	bool pgdownPressed = false; // Page Down = save network

	size_t trainingOffset;

	// Load and train the network before we begin
	std::vector<std::vector<float>> files;
	std::vector<uint32_t> fileIndices;
	//loadTrainingFiles(filenames, MY_PATH);
	//doTrainingSamples(compute, network, MY_PATH);

	while (!window.shouldClose()) {
		keyDown = 0;
		window.getSize(&width, &height);
		window.getCam().updateProjectionView(width, height, 800.f, oglopp::Camera::ORTHO);

		network.feedForward(compute);

		Layer* output = &network.getLayers().back();
		Neuron* neurons = static_cast<Neuron*>(output->getNeurons().map(oglopp::SSBO::BOTH));
		for (int i=0;i<10;i++) {
		 	neurons[i].expected = window.keyPressed(GLFW_KEY_0 + i) ? 1.0 : 0.0;
			keyDown = (neurons[i].expected > 0) ? GLFW_KEY_0 + i : keyDown;
		}
		for (int i=0;i<26;i++) {
			neurons[i + 10].expected = window.keyPressed(GLFW_KEY_A + i) ? 1.0 : 0.0;
			keyDown = (neurons[i + 10].expected > 0) ? GLFW_KEY_A + i : keyDown;
		}
		output->getNeurons().unmap();

		if (keyDown > 0 && !justPressed) {
			justPressed = true;
			saveTrainingElement(network.getLayers().front().getNeurons(), keyDown, MY_PATH);

			network.backProp(compute);

			output = &network.getLayers().front();
			Neuron* neurons = static_cast<Neuron*>(output->getNeurons().map(oglopp::SSBO::BOTH));

			for (size_t i=0;i<output->getNeurons().getSize()/sizeof(Neuron);i++) {
				neurons[i].expected = 0.0;
				neurons[i].value = 0.0;
			}

			output->getNeurons().unmap();
		}

		if (keyDown == 0) {
			justPressed = false;
		}


		if (window.keyPressed(GLFW_KEY_ENTER)) {
			if (enterPressed == false) {
				// Enter was just pressed
				trainingOffset = 0;
				trainingToggle = !trainingToggle;
				if (trainingToggle) {
					loadTrainingFiles(files, fileIndices, MY_PATH);
				}
			}
			enterPressed = true;

		} else {
			enterPressed = false;
		}

		if (trainingToggle) {
			doSomeSamples(compute, network, MY_PATH, files, fileIndices, trainingOffset, TRAINSIZE);
		}


		// Saving the network
		if (window.keyPressed(GLFW_KEY_PAGE_DOWN)) {
			if (pgdownPressed == false) {
				network.save(MY_PATH + MODEL_DIRECTORY);
			}
			pgdownPressed = true;
		} else {
			pgdownPressed = false;
		}


		//if (window.keyPressed(GLFW_KEY_RIGHT_ALT)) {
		//	network.backProp(compute);
		//}


		window.clear();

		shader.use();
		shader.setVec2("cursor", window.getCursorPos());
		shader.setBool("lalt", window.keyPressed(GLFW_KEY_LEFT_ALT));
		shader.setBool("lctrl", window.keyPressed(GLFW_KEY_LEFT_CONTROL));
		shader.setBool("leftClick", window.mousePressed(GLFW_MOUSE_BUTTON_LEFT));
		shader.setBool("rightClick", window.mousePressed(GLFW_MOUSE_BUTTON_RIGHT));
		shader.setVec2("resolution", glm::vec2(width, height));
		shader.setFloat("drawSize", InputBuffer::drawSize);
		network.draw(window, shader);
		//screen.draw(window, &shader);
		SSBO::unbind();

		window.bufferSwap();
		window.pollEvents();
	}

	return 0;
}
