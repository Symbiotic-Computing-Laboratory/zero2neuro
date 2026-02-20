[Base Index](../../index.md)  
[Previous Index](../index.md)  
# Network Builder

TODO: intro material

## Description

The Network Builder module is in charge of creating the model's
network. It does this by taking in user-defined arguments and using
keras3_tools to create either fully connected networks or
convolutional neural networks.  

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

### Regression vs Classification
The two biggest types oft asks for neural networks are regression and classification.  
#### Regression
- Predicts specific values
- Used in cases such as stock price prediction
- Output layer often has a linear activation function
#### Classification
- Predicts categories
- Example would be spam vs non-spam emails
- Binary Classification (two cases): sigmoid activation
- Multi-class Classifcation (n cases): softmax activation


## Zero2Neuro-Supported Network Types

- [Fully-Connected Neural Network](fully_connected.md): Translate a
vector of values into another vector of values

- [Convolutional Neural Network](cnn.md): Translate 1D (timeseries),
2D (images), or 3D (volumetric) data into a vector of values

- Recurrent Neural Network: Process sequences by taking in information from previous steps