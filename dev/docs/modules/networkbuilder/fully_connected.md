[Base Index](../../index.md)  
[Previous Index](index.md)  
# Fully Connected Neural Network

## Introduction

A fully connected neural network (FCNN) generally translates one vector (the
inputs) into another vector (the outputs). Each neuron in a FCNN is connect to every neuron in the next layer which allows the model to learn more complex patterns in the data. 

## Key Components

### Layers:  
- Input: Is the layer that takes in data to give to the hidden layer.  
- Hidden: Is where computations happen with linear regression and nonlinear transformations, often thought of as a black box.  
- Output: Is data that the model predicts, there are also nonlinear transformations that occur between the hidden and otuput layers. This can also contain predetermined values while training a model.

### Activation Functions:  
Activation functions are what introduct nonlinearity into the the model. A few examples are relu, sigmoid, and tanh. For a full list see [Keras Activation Function Documentation](https://keras.io/api/layers/activations/).

### Input Shape
The input shape is the dimension of the data and is one of the few things that one must specify before an experiment. When preparing the data for use it is turned into a tensor, which can be interpreted as an n-dimensional array. This is declared in the `network_config.txt` file under a `--input_shape` argument.  
Example: for a csv file that has 8 features  
```---input_shape 8```

### Hidden Layers
The hidden layer is responsible for taking the input data and doing linear regression, but it also applies non-linearity via the activation functions. The hidden layer also utilizes weights and will adjust these weights through a process called back-propagation. There are no strict guidelines for what a hidden layer should look like, but it is generally best to start small and build up in complexity. Here is an example of a small two layer network described in the `network_config.txt` file.
```
--number_hidden_units  
10  
5
```
This produces a neural network with ten neurons in the first layer and five in the second. 

### Output Layer Shape

The output layer shape depends on what the task is for example,
- Regression: A single neuron that has a linear activation function
- Binary Classification: Single neuron that has a sigmoid activation function
- Multi-class Classification: n Neurons where n is the number of classes with a softmax activation function
  
___

## Example Network Configuration
```
--network_type=fully_connected
--input_shape 
8
--number_hidden_units
10
5
--output_shape
1
--output_activation=linear

```
___

## Oddities

Dropout can be applied to both input and hidden layers in order to reduce overfitting.  
Output shape can be multi-dimensional if it's appropriate for the task for instance multi task learning.