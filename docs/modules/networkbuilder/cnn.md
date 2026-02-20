[Base Index](../../index.md)  
[Previous Index](index.md)  
# Convolutional Neural Network

## Introduction

A convolutional neural network generally translates 1D, 2D, or 3D data
(the inputs) into a vector (the outputs)

## Key Components

### Convolutional Modules
A convolutional module is made of:
- Convolutional layer
- Pooling

#### Convolutional Layer
This layer applies learnable filters, called kernels, over the input.

**Kernel Size**  
The size of the kernel determines what the dimensions of the filter are (3x3, 5x5, ..., nxn)
- Small kernels will capture more detail
- Large kernels capture the larger patterns

**Number of filters**   
The number of filters will determine how many feature maps the layer will use.
- More filters means more capacity for complex patterns
- The downside is higher computational cost
Each filter learns to detect specific patterns like edges, shapes, etc.

**Strides**  
A stride is how far the filter will move across the input.
- Stride = 1: Filter takes one step through data, for images this would be one pixel.
- Stride > 1: Filter takes n steps through data and downsample the feature maps, can decrease compuational cost

**Max Pooling**  
Max pooling reduces the dimensions by finding the maximum value inside of certain section.
- Downsample feature maps
- Reduces computation

#### Global Max Pooling
Global max pooling reduces every feature map to a single value by taking the maximum value from the entire dimension of the feature map.

A 10x10 feature map gets reduced to 1x1 

Usually used before the final layer and helps prevent overfitting.

### Fully Connected Module
After the convolutional layers CNNs will often have fully connected layers.

TODO

### Output Shape

TODO
___

## Example CNN Configuration
TODO
