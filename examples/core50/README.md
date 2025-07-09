# Example: Detecting Objects in Images

## Data
- File: [images.csv](images.csv)

## Network 
TODO: Describe core50's network
- Takes in a 128x128 image with RGB color channels.

## Details
This is a convolutional neural network example, which is a type of deep neural network but has the capability of image processing. If you look inside of the data file you will also notice that there aren't images but rather a .csv of file paths. These images are in Dr. Fagg's directory on Schooner or can be found online by looking at the core50 dataset. The core50 dataset is made up of many images of objects and this model's goal is to be able to tell the difference between those everyday objects.

## Experiment Suggestion
Try training with multiple objects and see if the model stays accurate or starts confusing objects with one another.
