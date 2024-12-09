# Sketch-ML
A basic machine learning vision playground to create and test training data, and my third, most efficient Artificial Neural Network. 

## How do I use it?
Draw in the input box on the left. Left click will increase the values near the cursor, and right click will decrease the values near the cursor.
To save a training image and backpropagate the model once, press any number or letter on your keyboard. This will save a file in the samples directory with the input image as an array of 4-byte floats.

# Network
The artificial network is written from scratch utilizing an OpenGL compute shader to perform the forward pass and backpropagation. It also uses shader storage buffer objects for storing and transferring data to the GPU. 

## What can it do?
Sketch-ML was designed with only the ability to detect handwritten digits in mind. However the ouput layer contains 36 outputs, enough for every letter of the english alphabet and arabic numeral digit. 
<br>
The output layer found below the large input box, starts at index 0 on the right and index 35 on the left (I'm sorry). In the below screenshots, we can see a successful example where digits were recognized successfully. These digits are all recognized on the same model with the same training set I created myself, which you can [download from my website](https://cdn2.honeybeeks.net/fa977edcb62e.tgz). 
<br>
This example was performed using a single hidden layer of size 100x100 (10000) neurons which is displayed on the right in each screenshot. The cost at each neuron is shown in blue in the same graphic. 
![image](https://github.com/user-attachments/assets/4034495d-c95c-4e38-a4ef-fbb2935f95bc)
![image](https://github.com/user-attachments/assets/a895526d-f307-4cac-87bb-af24d4391354)
![image](https://github.com/user-attachments/assets/0d3b280f-7045-47a5-9c50-42c7419d7454)
![image](https://github.com/user-attachments/assets/8c6a211e-38a7-4cb1-a927-e4943cadf4b7)
![image](https://github.com/user-attachments/assets/770d03f3-8398-4424-be14-1cac7b9acbcd)
![image](https://github.com/user-attachments/assets/530321a5-521a-40c5-8951-0eaa87c1dfed)
![image](https://github.com/user-attachments/assets/a96585fe-d831-4cf3-a129-48e594ab01a9)
![image](https://github.com/user-attachments/assets/a95e7adb-9277-45e4-b0a8-64902313a0f8)
![image](https://github.com/user-attachments/assets/a5a76c63-5bc4-47dc-9040-1d0649c34116)
![image](https://github.com/user-attachments/assets/2caee386-684e-40aa-99b4-aff4dca40e62)
