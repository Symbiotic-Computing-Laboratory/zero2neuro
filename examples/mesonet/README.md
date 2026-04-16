---
title: Daily Rainfall Retrieval
nav_order: 40
has_children: false
parent: Zero2Neuro Examples
---

# Daily Rainfall Prediction using Oklahoma Mesonet Data  

The data used in this example comes from the Oklahoma Mesonet, from 1994-2000, which is a network of enviromental monitoring stations around the state of Oklahoma. These monitoring stations provide high quality enviromental measurements and the goal of this example is to use those variables (e.g., temperature, humidity, dewpoint, wind speed) to predict rainfall.
  
We approach this problem in two different ways:  
1. **Regression**: Predict the exact rainfall in inches
2. **Classification**: Predict whether it rained or not  

Modeling like this is popular in both meterology and agriculture where rainfall predictions could benefit flood watches/warnings and crop managment. 

---

## Data  

- [Source](dataset_source.md)
- [Regression Dataset](data/regression_meso_data.csv)
- [Classification Dataset](data/categorical_meso_data.csv)

The dataset includes:
1. Time (YEAR/MONTH/DAY)
2. Enviromental variables
3. Binary rainfall indicator, classification only (RAINED, NO_RAIN)

---

## Data Configuration

- [Data Configuration File](data_config.txt)

Note:
- Regression only:  All inputs are **numerical**, so no tokenization or embedding is required
- This is a subsampled and cleaned up dataset, the notebook for cleaning the dataset is included. 

---

## Prediction with Enviromental Data

Enviromental data is both **temporal and spatial** as the data comes from different times and locations. The relationships between certain variables, like dew point and atmospheric pressure, can influence rainfall. 

Unlike sequenced data this dataset is made of indepedent numerical values. However, this problem can be rearranged to take into account prior weather via the data variables.

--- 

## Networks

We've provided a fully connected architecture for regression and classification:  
  
  - **Fully Connected Network**
    - A standard deep neural network using dense layers to learn relationships between the enviromental variables
    - Is by default regression but can be made into a classification model by passing in the [classification override file](classification_override.txt) after the three argument files
  
---

## Notes

This example helps highlight how a single dataset can support **multiple machine learning tasks**
- Predicting exact rainfall (regression)
- Predicting rainfall occurence (classification)
