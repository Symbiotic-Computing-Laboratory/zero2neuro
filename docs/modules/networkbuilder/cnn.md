---
title: Convolutional Neural Networks
nav_order: 30
parent: Network Builder
has_children: true
---

# Convolutional Neural Networks

In many situations, the input to a model involves data that has some temporal or spatial organization.  For example:
- Timeseries data containing environmental features, such as temperature and pressure.  In this 1-dimensiaonl case, the input data is in the shape of a matrix (T,2), where T is the number of timesteps in the example, and the two _channels_ are temperature and pressure.
- Images that have pixels organized along both height and width, with each pixel containing information about the color of the pixel in terms of red, green, and blue magnitudes.  For this 2-dimensional case, the shape of the input data is (R,C,3), where R is the number of image rows; C is the number of columns; and the 3 corresponds to the red/green/blue channels.
- Atmospheric state can be described as small, 3D volumes of air (voxels), each of which have features of temperature, pressure, water content, and velocity.  The shape of the input data is (X,Y,Z,F), where X, Y, Z are the extents along the cardinal directions, and F corresponds to the number of features.

Convolutional Neural Networks (CNNs) are particularly powerful in their ability to identify local patterns in the input data at many different scales.  For a CNN that takes an image as an input, the network might first identify small edges in the image, then combine those ediges into larger scale patterns, ultimately resulting in pattern detectors for high-level features, such as eyes, beaks, feathers, and talons.  

## Convolutional Modules
In Zero2Neuro, a CNN is implemented as a stack of _convolutional modules_.  Each module contains a sequence of neural layers; at minimum a module contains:

1. __Convolutional Layer__: responsible for identifying spatial patterns over some limited region of the input.   This layer "searches" for each spatial pattern over the entire input.  These layers are defined by:
   - The size of the temporal/spatial region.  For images, this is typically 3x3 or 5x5 pixels
   - The number of distinct patterns to identify.  This is referred to as the _number of filters_ or _channels_
   - A non-linear activation function

The output of this layer also contains temporal/spatial extent, though it may be smaller in size than the input.

2. __Max Pooling Layer__: computes the maximum over small temporal/spatial regions.  This is defined by:
   - The size of the region.  For images, these regions are typically 2x2
   
In Zero2Neuro, the max pooling layer also reduces the size of the output commensurate with the size of the pool.  So, each 2x2 pixel region will produce 1 pixel as output.  Thus, a 2x2 pool will reduce produce an output that is half the size of the input in each of the number of rows and columns.  

One way to interpret the combination of these two layers is that the convolutional layer identifies patterns from the input, and then the max pooling layer asks whether the patterns _exist_ over the small region.

___
## Example Network

### Image Inputs
In this example, the input to the network is a single 3-color image. The input shape must reflect the image size:

```
--network_type=cnn

--input_shape
128
128
3
```

### Convolutional Modules
These modules are specified using several arguments.  Here, we define a sequence of four modules
```
--conv_kernel_size
3
3
5
5

--conv_number_filters
8
16
16
32
--conv_activation=elu

--conv_pool_size
0
2
0
0
```

Configuration notes:
- Each module has a corresponding element from each argument
- ```--conv_kernel_size``` specifies the size of the filters.  Because this is a 2D problem, then the 3s and 5s are automatically translated into 3x3 and 5x5 kernels, respectively.
- ```--conv_number_filters``` specifies the number of filters for each model.  It is typical that this number increases in the deeper modules.
- ```--conv_activation``` is the activation function used for each filter.
- ```--conv_pool_size``` defines the size of the pools for the max pooling step.  Here, only the second module involves a degree of pooling (0 = no pooling)

Convolutional modules:
- Module 0: eight 3x3 filters with no pooling
- Module 1: sixteen 3x3 filters with pooling.  This reduces the size of the image by a factor of 2 in both the rows and columns
- Module 2: sixteen 5x5 filters
- Module 3: 32 5x5 filters

### Global Max Pooling

Global max pooling reduces every feature map to a single value by taking the maximum value across the extent of each feature.  In this example, the TODO

Usually used before the final layer and helps prevent overfitting.

### Fully Connected Module
After the convolutional layers CNNs will often have fully connected layers to make the final predictions.

### Output Shape
The output shape will depend on the task.
Classification: Number of neurons isequal to the number of classes. Activation function will be either softmax or sigmoid.
Regression: A single neuron with a linear activation function.

### Input shape
The input shape must match what the dimensions of the model are expected to be.
-1D data: (batch_size, length, channels)
-2D data: (batch_size, height, width, channels)
-3D data: (batch_size, depth, height, width, channels)
___

## Example CNN Configuration
```
--network_type=cnn
--input_shape
128
128
3
--conv_kernel_size
3
3
5
5
--conv_padding=valid
--conv_number_filters
8
16
16
32
--conv_activation=elu
--conv_pool_size
0
2
0
2
--number_hidden_units
20
10
--hidden_activation=elu
--output_shape
2
--output_activation=softmax
```
### Explanation: 
- Four convolutional layers that have kernels of 3x3, 3x3, 5x5, 5x5 with 8, 16, 16, 32 filters in each respectively.
- Max pooling is applied to the second and fourth convolutional layer with a pool size of 2x2
- Fully connected network at end with a layer of 20 neurons and a layer of 10 neurons with an elu activation function.
- Output layer has a size of 2 (model is predicting between two classes) and a softmax activation function.

## Oddities
- Padding can affect the dimensions of the output (valid or same)
