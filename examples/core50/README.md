---
title: Image Classification
nav_order: 70
has_children: false
parent: Zero2Neuro Examples
---

# Example: Detecting Objects in Images

## Overview
This is a convolutional neural network image classification example.  Here, we are using a small subset of the [Core 50](https://vlomonaco.github.io/core50/) image dataset. The data configuration file describes the full list of images to be loaded and their class labels.  The images are not included as a part of this repository; there are two options:
- Download the 128x128 dataset from [Core 50](https://vlomonaco.github.io/core50/)
- If you are on the OU Supercomputer, then the images can be accessed from: ~fagg/datasets/core50 

## Data
- This example contains only a single data set that is used for training (i.e., no validation or testing data sets)
- Configuration: [images.csv](images.csv)

## Network 
- Takes in a 128x128 image with RGB color channels and classifies the image as being in one of two classes
- The network architecture is a simple illustration of what can be implemented:
   - 4 convolutional layers, with every other one followed with a max pooling step
   - A Global Max Pool step (implicit) reduces the feature arrays down to indivdual scalars
   - 2 hidden layers combine the features together in arbitrary ways
   - THe output is 2 units (one for each class)
   - Network description: [network.txt](network.txt)

## Experiment
- Loss: sparse categorical cross-entropy.  This matches the fact that Class in the data file is an integer representation of the true class label.
- Merics: sparse categorical accuracy
- Configuration: [experiment.txt](experiment.txt)

## Experiment Suggestion
Try training with multiple objects and see if the model stays accurate or starts confusing the objects with one another.
