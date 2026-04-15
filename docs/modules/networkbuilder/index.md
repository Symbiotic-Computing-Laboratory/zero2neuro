---
title: Network Builder
nav_order: 20
parent: Zero2Neuro Modules
has_children: true
---
# Network Builder

The _Network Builder_ module is responsible for creating the specific neural network model for your experiments.  The details of the network type (schema), the number and types of layers, and nature of the network output are all specified using arguments within a configuration file.

## Modeling Problem Types
Deep neural networks solve a range of different problem types, determined by the output of the model and how that output is interpretted.  

determined by the model output, the training data desired outputs, and the loss fun

.  Two of the key types are 

Two key modeling problem types 

Regression vs Classification
The two biggest types of tasks for neural networks are regression and classification.  
#### Regression
- Predicts specific values
- Used in cases such as stock price prediction
- Output layer often has a linear activation function
#### Classification
- Predicts categories
- Example would be spam vs non-spam emails
- Binary Classification (two cases): sigmoid activation
- Multi-class Classifcation (n cases): softmax activation

##
- [Introduction to Neural Networks](network_tutorial.md) (Tutorial)
- Network Schemata 
   - [Fully-Connected Neural Network](fully_connected.md): Translate a
vector of values into another vector of values (Beginner)
   - [Convolutional Neural Network](cnn.md): Translate spatial and/or temporal data (Intermediate)

Translate 1D (timeseries),
2D (images), or 3D (volumetric) data into a vector of values

   - [Recurrent Neural Network](rnn.md): Process sequences by taking in information from previous steps  (Intermediate)
- 


