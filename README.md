# Sketch-ML - More-Layers Branch
An adaptation of Sketch-ML with added support for convolutional layers, pooling layers, and eventually more, in addition to the existing fully-connected layers.

## How do I use it?
Draw in the input box on the left. Left click will increase the values near the cursor, and right click will decrease the values near the cursor.
To save a training image and backpropagate the model once, press any number or letter on your keyboard. This will save a file in the samples directory with the input image as an array of 4-byte floats.

# Network
This deep neural network is written from scratch utilizing an OpenGL compute shader to perform the forward pass and backpropagation. It also uses shader storage buffer objects for storing and sending data to the GPU.

## What can it do?
Sketch-ML [MoLa] is designed to be able to allow the user to specify a variable number of layers, with variable types at runtime. These layers are loadable from a model file as well, allowing models to be shared and loaded easily later on.
