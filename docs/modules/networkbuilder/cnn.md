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
- Atmospheric state can be described as small, 3D volumes of air (voxels), each of which have features of temperature, pressure, water content, and wind velocity.  The shape of the input data is (X,Y,Z,F), where X, Y, Z are the extents along the cardinal directions, and F corresponds to the number of features.

Convolutional Neural Networks (CNNs) are particularly powerful in their ability to identify local temporal or spatial patterns in the input data at many different scales.  For a CNN that takes an image of an animal as an input, the network might first identify small edges in the image, then combine those edges into larger scale patterns, ultimately resulting in pattern detectors for high-level features, such as eyes, beaks, feathers, and talons.  

## Convolutional Modules
In Zero2Neuro, a CNN is implemented as a stack of _convolutional modules_.  Each module contains a sequence of neural layers; at minimum a module contains:

__Convolutional Layer__: responsible for identifying spatial patterns over some limited region of the input.   This layer "searches" for each spatial pattern over the entire input.  These layers are defined by:
   - The size of the temporal/spatial region.  The details of the specific pattern are encoded in a trainable __kernel__.  For images, kernels are typically 3x3 or 5x5 pixels.
   - The number of distinct patterns to identify.  This is referred to as the _number of filters_ or _channels_
   - A non-linear activation function.

The output of this layer also contains temporal/spatial extent, though it may be smaller in size than the input.

__Max Pooling Layer__: computes the maximum over small temporal/spatial regions.  This is defined by:
   - The size of the region.  For images, these regions are typically 2x2
   
In Zero2Neuro, the max pooling layer also reduces the size of the output commensurate with the size of the pool.  So, each 2x2 pixel region will produce 1 pixel as output.  Thus, a 2x2 pool will reduce produce an output that is half the size of the input in each of the number of rows and columns.  

One way to interpret the combination of these two layers is that the convolutional layer identifies patterns from the input, and then the max pooling layer asks whether the patterns _exist_ over the small region.

___
## Example Network

### Image Inputs
In this example, the input to the network is a single 3-color image that has 128 rows and columns. The input shape must reflect this size:

```
--network_type=cnn

--input_shape
128
128
3
```

### Convolutional Modules
These modules are specified using several arguments.  Here, we define a sequence of four modules:
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
- ```--conv_kernel_size``` specifies the size of the kernels.  Because this is a 2D problem, then the 3s and 5s are automatically translated into 3x3 and 5x5 pixel kernels, respectively.
- ```--conv_number_filters``` specifies the number of filters for each module.  It is typical that this number increases in the deeper modules.
- ```--conv_activation``` is the activation function used for each filter.
- ```--conv_pool_size``` defines the size of the pools for the max pooling step.  Here, only the second module involves a degree of pooling (0 = no pooling)

The four convolutional modules:
- Module 0: eight 3x3 kernels with no pooling.  Early layers in an image CNN are largely concerned with identifying edge-like features of different colors.
- Module 1: sixteen 3x3 kernels __with pooling__.  This reduces the size of the image by a factor of 2 in both the rows and columns
- Module 2: sixteen 5x5 kernels.  The larger kernel size allows the CNN to identify more complex patterns.
- Module 3: 32 5x5 kernels.

### Global Max Pooling

Global max pooling reduces every feature map to a single value by taking the maximum value across the full extent of the spatial domain.  In this example, the output of the last convolutional module still contains a spatial representation for each of 32 features with some number of rows and columns.  If feature 5 recognizes _eyes_ at a specific location, then one can interpret global max pooling as summarizing the feature map in terms of _there exists an eye somewhere in the image_.


### Fully-Connected Stack
Typically, the identified features (32 in this example) must then be combined together to determine the output of the network.  This is accomplished through the use of a stack of fully-connected modules that are defined using the same arguments that are used for [Fully-Connected Networks](fully_connected.md).

In this example, our CNN is responsible for classifying images into one of two categories.

```
--number_hidden_units
20
10
--hidden_activation=elu

--output_shape
2
--output_activation=softmax
```

Configuration notes:
- The single value specified by ``--output_shape`` determines the number of classes
- ```--output_activation`` of _softmax_ produces as output a probability distribution over the two classes.  This choice is appropriate when the classes are _exclusive_ (the true class can be exactly one, but not more)

## Fully-Connected Neural Network Examples
- Classification: 
   - [Image Classification](../../../examples/core50/README.md)
- Classification and Regression: 
   - [Peptide Binding](../../../examples/amino/README.md) (see the CNN implementation)

## More Details (Intermediate)
- [Convolutional Neural Network Details](cnn_details.md)
