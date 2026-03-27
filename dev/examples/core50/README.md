# Example: Detecting Objects in Images

## Overview
This is a convolutional neural network example, which is a type of deep neural network but has the capability of image processing. If you look inside of the data file you will also notice that there aren't images but rather a .csv of file paths. These images are in ~fagg/datasets/core50 on Schooner or can be found online by looking at the core50 dataset. The core50 dataset is made up of many images of objects and this model's goal is to be able to tell the difference between those everyday objects. 

## Data
- This example contains only a single data set that is used for training (i.e., no validation or testing data sets)
- Configuration: [images.csv](images.csv)

## Network 
- Takes in a 128x128 image with RGB color channels and classifies the image as being in one of two classes
- The network architecture is a simple illustration of what can be implemented:
   - 4 convolutional layers, with every other one followed with a max pooling step
   - A Global Max Pool step (implicit) reduces the feature arrays down to indivdual scalars
   - 2 hidden layers combine the features together in arbitrary ways
   - Output is 2 units (one for each class
   - Network description: [network.txt](network.txt)

## Experiment
- Loss: sparse categorical cross-entropy.  This matches the fact that Class in the data file is an integer representation of the true class label.
- Merics: sparse categorical accuracy
- Configuration: [experiment.txt](experiment.txt)

## Experiment Suggestion
Try training with multiple objects and see if the model stays accurate or starts confusing objects with one another.
