---
title: Network Builder
nav_order: 20
parent: Zero2Neuro Modules
has_children: true
---
# Network Builder

The _Network Builder_ module is responsible for creating the specific neural network model for your experiments.  The details of the network type (schema), the number and types of layers, and nature of the network output are all specified using arguments within a configuration file (typically the _network_ configuration file).

## Prediction Problem Types
Deep neural networks solve a range of different problem types, determined by the:
1. the output of the model and how it is interpretted, 
2. how the desired outputs are expressed, and
3. the _loss_ function that is used to compare model outputs to the desired outputs

The fundamental types of prediction problem supported by Zero2Neuro are _Regression_ and _Classification_.  

### Regression
- Output type: a real (continuous) value, a vector of real values, or any shaped tensor of values
- Output interpretation: any continuous prediction, such as rainfall rate, or stock price
- Desired outputs: also real values
- Loss function: it is not uncommon to use _Mean Squared Error (mse)_ or _Mean Absolute Error (mae)_

### Binary Classification
- Output type: a continous value in the range 0..1 (or any shaped tensor)
- Output interpretation: the probability of some event, such as the probability that a protein will bind to some molecule, or the probability that an image contains a cat
- Desired outputs: binary values (0 or 1)
- Loss function: _binary cross-entropy_

### Categorical Classification
- Output type: a vector of continuous values that 1) each fall in the range 0..1, and 2) sum to 1
- Output interpretation: probability distribution over a discrete set of choices
- Desired outputs: two typical options:
   1. Categorical Classification: One-hot encoded representation of the correct choice (all zeros, except for a single one in the correct position)
   2. Sparse Categorical Classification: a single natural integer (0, 1, 2, ...) that corresponds to the correct answer
- Loss function: two options:
   1. _categrocial cross-entropy_
   2. _sparse categorical cross-entropy_
 
## Network Architectures Types
Different architecture types apply to different forms of model inputs and model outputs.  Zero2Neuro includes schemata for a number of common network architecture types.  Each architecture type supports regression and classification problems, depending on how it is configured.  The key network types are:

1. __Fully-Connected Networks__: 
   - The input is typically a continuous vector of some size
   - The output is also some scalar value or vector

2. __Convolutional Neural Networks (CNNs)__:
   - The input contains some form of spatial or temporal structure, including:
      - A a squence of vectors (a 1-dimensional input)
         - Example: the position and velocity of some object over time
         - Example: a sequence of words
      - A 2D grid of vectors (a 2-dimensional input)
         - Example: a color image
      - A 3D grid of vectors (a 3-dimensional input)
         - Example: a description of the atmosphere in terms of a vector at each 3D grid cell (e.g., the temperature, pressure, and speed of each voxel)
    - The output is some scalar value or vector

3. __Recurrent Neural Networks (RNNs)__:
   - The input contains some 1D spatial or temporal structure
   - The output can be either:
      - A scalar value or vector
      - A 1D sequence of scalar values or vectors (often, there is one output vector for each input vector)

## More Details
- [Introduction to Neural Networks](network_tutorial.md) (Tutorial)
- Network Schemata 
   - [Fully-Connected Neural Network](fully_connected.md) (Beginner)
   - [Convolutional Neural Network](cnn.md) (Intermediate)
   - [Recurrent Neural Network](rnn.md) (Intermediate)
- TODO: tokenization and embedding
- TODO: preprocessing


