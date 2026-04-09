---
title: Wisconsin Diagnostic Breast Cancer (WDBC)
nav_order: 30
parent: Zero2Neuro Examples
---
# Example: Detecting Malignant or Benign Breast Tumors

## Data: Wisconsin Diagnostic Breast Cancer (WDBC)
- Data file: [wdbc.csv](wdbc.csv)
- Source information: [wdbc.txt](wdbc.txt)
- [data_config.txt](data_config.txt)

## Network
- Binary Classification 
- Twenty-nine inputs (Details on tumor)
- Two Hidden Layers
- One output (M or B, Binary)
- [network_config.txt](network_config.txt)

## Training Details
- [experiment_config.txt](experiment_config.txt)

## Details
This model takes in the descriptive details of a tumor and
predicts if it is benign or malignant. It uses binary classification
to give a prediction of confidence between the tumor being malignant
or benign. The output function is a sigmoid function which is
contained within the 0-1 range, which can be interpretted as a probability.

## Experiment Suggestion
Try to limit the set of input features to identify if some are more
more important to the prediction process than others.  
