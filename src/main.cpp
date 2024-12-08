#include <cstddef>
#include <oglopp.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cmath>
#include <filesystem>

#include "network.h"
#include "oglopp/camera.h"
#include "oglopp/more_shapes.h"
#include "oglopp/ssbo.h"

using namespace oglopp;

#define RESOLUTION	(32) // RESOLUTION x RESOLUTION pixels
#define PIXELS		(RESOLUTION * RESOLUTION)

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

int main() {
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

	// Create a list of vertices with just an ID. the position will be provided in an SSBO calculated by a compute shader
	Rectangle screen;
	screen.setScale(glm::vec3(1.0, 1.0, 1.0));
	screen.setPosition(glm::vec3(0.0, 0.0, 1.0));

	// Create a network
	Network network(32*32, {16*16 , 16*16}, 36);

	int width, height;
	//window.getCam().aimBy(0, 0);
///	window.getCam().setTarget(window.getCam().getAngle());

	while (!window.shouldClose()) {
// #if 0
// 		ParticleInfo* result = static_cast<ParticleInfo*>(ssbo.map());
// 		if (result == nullptr) {
// 			std::cerr << "Uhh.. the map pointer was null? That can happen?" << std::endl;
// 			return -1;
// 		}

// 		ssbo.unmap();
// #endif
//
//
		//window.handleNoclip();

		window.getSize(&width, &height);
		window.getCam().updateProjectionView(width, height, 800.f, oglopp::Camera::ORTHO);

		network.feedForward(compute);
		if (window.keyPressed(GLFW_KEY_RIGHT_ALT)) {
			network.backProp(compute);
		}

		window.clear();

		shader.use();
		shader.setVec2("cursor", window.getCursorPos());
		shader.setBool("lalt", window.keyPressed(GLFW_KEY_LEFT_ALT));
		shader.setBool("lctrl", window.keyPressed(GLFW_KEY_LEFT_CONTROL));
		shader.setBool("leftClick", window.mousePressed(GLFW_MOUSE_BUTTON_LEFT));
		shader.setBool("rightClick", window.mousePressed(GLFW_MOUSE_BUTTON_RIGHT));
		shader.setVec2("resolution", glm::vec2(width, height));
//		shader.setVec2("layerSize", glm::vec2(RESOLUTION, RESOLUTION));
		shader.setFloat("drawSize", InputBuffer::drawSize);
		network.draw(window, shader);
		//screen.draw(window, &shader);
		SSBO::unbind();

		window.bufferSwap();
		window.pollEvents();
	}

	return 0;
}
