---
title: Neural Networks Tutorial
nav_order: 10
parent: Network Builder
has_children: true
---
TODO

## Key Concepts

### Individual neurons

A neuron is a computational unit of a neural network.  

Each neuron:  
- Take n inputs
- Multiply each input by a corresponding weight
- Add a bias to the weighted sum of inputs
- Apply a non-linear activation

### Non-linearities
Non-linear activation functions are what make neural networks special, without non-linearity a model can only graph a linear function.  
  
Activation function examples:
- ReLU
- Tanh
- Sigmoid
- Softmax

These non linear activation functions allow neural networks to:
- Learn complex relationships
- Approximate non-linear functions

### Layers
Layers are collections of neurons inside the section of the network.

There are three types of layers:
- Input layer: Takes in input data
- Hidden layers: Perform transformations on input data
- Output layers: Makes the final prediction

Each layer transforms its input and deep neural networks can have several layers.

### Network Architecture
A model's architecture consists of:
- Number of layers 
- Number of neurons in each layer
- How each layer connects

The more complex an architecture the better its ability to approximate complex functions.

Architecture Examples:
- Fully connected networks
- Convolutional networks
- Recurrent networks

### Propagation 
#### Forward Propagation
Forward propagation is the process than eables passing input data through each layer.

Each layer:
- The linear transformation gets applied (weight and bias)
- Non-linear activation function is applied
- Result is passed to the next layer

Once it reaches the output layer the model will make a prediction.
#### Backpropagation
TODO

### Loss Functions
Loss functions show the level of error between the prediction and true value.

Examples:  
- Mean Squared Error  
- Mean Absolute Error
- Cross-Entropy Loss 

### Optimizers 
Optimizers update the model parameters in order to minimize the loss function.

Examples:
- Adam
- Lion
