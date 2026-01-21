[Base Index](../../index.md)  
[Previous Index](index.md)  
# Fully Connected Neural Network

## Introduction

A fully connected neural network generally translates one vector (the
inputs) into another vector (the outputs).

## Key Components

Three types of layers:  
- Input: Is the layer that takes in data to give to the hidden layer.  
- Hidden: Is where computations happen with linear regression and nonlinear transformations, often thought of as a black box.  
- Output: Is data that the model predicts, there are also nonlinear transformations that occur between the hidden and otuput layers. This can also contain predetermined values while training a model.

Activation Functions:  
There are several activation functions, for more details on the various ones you can use see [Keras Activation Function Documentation](https://keras.io/api/layers/activations/).

### Input Shape
The input shape is the dimension of the data and is one of the few things that one must specify before an experiment. When preparing the data for use it is turned into a tensor, which can be interpreted as an n-dimensional array. This is declared in the `network_config.txt` file under a `--input_shape` argument. As an example, for a csv file with 8 features, one would put 8 as the input shape.

### Hidden Layers
The hidden layer is the black box of the model and is responsible for taking the input data and doing linear regression, but it also applies non-linearity via the activation functions. The hidden layer also utilizes weights and will adjust these weights through a process called back-propagation. There are no strict guidelines for what a hidden layer should look like, but it is generally best to start small and build up in complexity. Here is an example of a small two layer network described in the `network_config.txt` file.  
`--number_hidden_units`  
`10`  
`5`  
This produces a neural network with ten neurons in the first layer and five in the second. 

### Output Shape

non-linearity

## Regularization

Regularization techniques are used in machine learning to combat over fitting. They try to penalize the model for becoming too complicated and help move the model to being more simple and generalized to the problem. 

###  Dropout
Dropout is a regularization technique that "drops out" a fraction of the neurons in a network by setting their output to 0. This combats neurons becoming too dependent on each other and encourages the model to look at more features. Dropout uses a probability to "randomly" select neurons to disable during training. This technique is only used during training as it is best practice to have the full model active when you are going through a testing set. 

### L1 Regularization 

### L2 Regularization

## Batch Normalization

___

## Example Network Configuration

___

## Oddities

Dropout_input

Output shapes can be multi-dimensional
