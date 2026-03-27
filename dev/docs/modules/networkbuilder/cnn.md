[Base Index](../../index.md)  
[Previous Index](index.md)  
# Convolutional Neural Network

## Introduction

A convolutional neural network (CNN) generally translates 1D, 2D, or 3D data
(the inputs) into a vector (the outputs). CNNs work by extracting spacial features using what are called convolutional filters and then downsample the data using "pooling" layers and then produce an output vector once downsampled enough.

## Key Components

### Convolutional Modules
A convolutional module is made of:
- Convolutional layer: Applies filters (kernels) across the input that can learn.
- Pooling: Reduces the spatial dimensions of the data while maintaining important features.

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
