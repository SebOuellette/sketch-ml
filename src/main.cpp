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
#include <unistd.h>

#include "defines.h"
#include "network.h"
#include "neuron.h"
#include "netutil.h"
#include "oglopp/camera.h"
#include "oglopp/compute.h"
#include "oglopp/more_shapes.h"
#include "oglopp/ssbo.h"

using namespace oglopp;

#define RESOLUTION	(32) // RESOLUTION x RESOLUTION pixels
#define PIXELS		(RESOLUTION * RESOLUTION)
#define TRAINSIZE 5
#define KEY_SAVE_MODEL	GLFW_KEY_PAGE_DOWN

#define OPT_STRING "h"

class InputBuffer {
public:
	static Window* windowPtr;
	static float drawSize;
	static void* keyData;

	enum INPUT_ACTION : int {
		KEY_PRESS = GLFW_PRESS,
		KEY_RELEASE = GLFW_RELEASE,
		KEY_REPEAT = GLFW_REPEAT
	};

	static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
		drawSize += yoffset * 3;
		drawSize = glm::max(drawSize, 1.f);
	}

	static void keyCallback(GLFWwindow* window, int key, int systemScanCode, int action, int mods) {
		//std::cout << "Pressed key " << key << " with action " << action << std::endl;
	}
};

Window* InputBuffer::windowPtr  = nullptr;
void* InputBuffer::keyData = nullptr;
float InputBuffer::drawSize = 30;


int main(int argc, char** argv) {
	srand(time(NULL));


	// Handle options
	int opt;
	while ((opt = getopt(argc, argv, OPT_STRING)) != -1) {
		switch(opt) {
			case 'h':
				std::cout << " SketchML v" << SKML_VERSION << " - Help Menu" << std::endl << std::endl
				<< "-h\t\tDisplay this help menu." << std::endl
				<< "-m [model.skm]\tSelect a relative or status path to load a model from." << std::endl
				<< "-s [sample dir]\tChoose the path where the dataset of samples can be located. " << std::endl
				<< "-t [type]\tOne of 'classify,deep'" << std::endl
				<< "-c [kernel]\tMake the previous hidden layer a convolutional layer. 'kernel' specifies which kernel containing a set of predefined filters to use." << std::endl
				<< "-L [neurons]\tAdd a new (hidden) layer of some size." << std::endl
				<< "-I [neurons]\tSpecify the number of neurons to use in the input layer." << std::endl
				<< "-O [neurons]\tSpecify the number of neurons to use in the output layer." << std::endl
				<< "-T [iterations]\tTrain the network for some number of 'iterations' then save the model and exit." << std::endl;
				exit(0); // Close the program after displaying help
				break;
		}
	}


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
	glfwSetKeyCallback(window.getWindow(), InputBuffer::keyCallback);

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
	Network network;
	std::string modelPath;
	if (argc < 2) {
		modelPath = MY_PATH + MODEL_DIRECTORY;
		network.setup(32*32, {50*50, 20*20, 16, 20*20, 50*50}, 32*32);
	} else {
		modelPath = "";
		network.setup(argv[1]);
	}

	int width, height;
	int8_t keyDown = 0;
	bool justPressed = false;
	bool enterPressed = false;
	bool trainingToggle = false;
	bool pgdownPressed = false; // Page Down = save network
	size_t trainingOffset = 0;

	// Load and train the network before we begin
	std::vector<std::vector<float>> files;
	std::vector<uint32_t> fileIndices;

	while (!window.shouldClose()) {
		keyDown = 0;
		window.getSize(&width, &height);
		window.getCam().updateProjectionView(width, height, 800.f, oglopp::Camera::ORTHO);

		network.feedForward(compute);

		Layer* output = &network.getLayers().back();
		//Neuron* neurons = static_cast<Neuron*>(output->getNeurons().map(oglopp::SSBO::BOTH));
		for (int i=0;i<10;i++) {
		 	//neurons[i].expected = window.keyPressed(GLFW_KEY_0 + i) ? 1.0 : 0.0;
			keyDown = window.keyPressed(GLFW_KEY_0 + i) ? GLFW_KEY_0 + i : keyDown;
		}
		for (int i=0;i<26;i++) {
			//neurons[i + 10].expected = window.keyPressed(GLFW_KEY_A + i) ? 1.0 : 0.0;
			keyDown = window.keyPressed(GLFW_KEY_A + i) ? GLFW_KEY_A + i : keyDown;
		}
		//output->getNeurons().unmap();

		if (keyDown > 0 && !justPressed) {
			justPressed = true;
			setExpectedOutput(network);
			std::cout << "pressed" << std::endl;
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
				network.save(modelPath);
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
