# Sketch-ML - AutoEncoder Branch
An adaptation of Sketch-ML with the GUI, network layers, and expected values set up to reproduce the input image.

## How do I use it?
Draw in the input box on the left. Left click will increase the values near the cursor, and right click will decrease the values near the cursor.
To save a training image and backpropagate the model once, press any number or letter on your keyboard. This will save a file in the samples directory with the input image as an array of 4-byte floats.

# Network
The artificial network is written from scratch utilizing an OpenGL compute shader to perform the forward pass and backpropagation. It also uses shader storage buffer objects for storing and transferring data to the GPU.

## What can it do?
Sketch-ML [AE] is designed to be able to automatically compress handwritten digits down to 10 floating point numbers.
This example was performed using 5 hidden layers of the following sizes: 50x50, 20x20, 10, 20x20, 50x50. The input and output layers are of size 32x32.
The following examples demonstrate autoencoding compression of handwritten digits which were not found in the training set.
![image](https://cdn2.honeybeeks.net/b9cdb599ea87.png)
![image](https://cdn2.honeybeeks.net/89f130cd5b13.png)
![image](https://cdn2.honeybeeks.net/7dfbc09fad43.png)
![image](https://cdn2.honeybeeks.net/9992481f48f.png)
![image](https://cdn2.honeybeeks.net/d811ea0648dd.png)
