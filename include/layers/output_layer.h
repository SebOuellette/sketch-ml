#ifndef OUTPUT_LAYER
#define OUTPUT_LAYER

#include "layer.h"

// Output Layer
class OutputLayer : public Layer {
public:
	/* @brief Setup a pooling layer
	 * @param[in] newNeuronCount	The new size of the input neurons in 3 dimensional space
	 * @return						A status code
	*/
	OutputLayer(uint64_t const& newNeuronCount);
};

#endif
