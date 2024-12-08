#ifndef NEURON_H
#define NEURON_H

#include <oglopp/ssbo.h>

struct Neuron {
	float bias;
	float value;
	float expected;
};


#endif
