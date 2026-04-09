---
title: Iris Classification
nav_order: 20
parent: Zero2Neuro Examples
---
# Example: Predicting Iris Flower Class

## Data
- File: [iris_data.csv](iris_data.csv)
- Citation: [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris)
- [data.txt](data.txt)

## Network
- Four input features
- Two hidden layers
- Three output units: one for each class, encoding the probability for
each class
- [network.txt](network.txt)

## Training Details
- [experiment.txt](experiment.txt)

## Details
This model predicts the type of iris based on four observed features.
The desired output is a single sparse categorical value (translated from a
string to a natural number)

## Experiment Suggestion
Try changing the hidden layers and numbers of units to see if the accuracy
changes

