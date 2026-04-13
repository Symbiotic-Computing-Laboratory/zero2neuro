---
title: Zero2Neuro Examples
nav_order: 30
has_children: true
parent: Zero2Neuro
---

# Zero2Neuro Examples

The following examples explore the essential features of Zero2Neuro,
inluding different data formats and network architectures.

- [Exclusive-Or](xor/README.md): simple 2-input, 1-output logic
problem.
   - Problem type: logistic regression
   - Input data: binary feature vector
   - Model type: fully-connected
   - Data file type: csv
   - Loss function: mean squared error

- [Iris](iris/README.md): UCI iris flower classification problem
   - Problem type: classification
   - Input data: feature vector
   - Model type: fully-connected
   - Data file type: csv
   - 4 input features
   - 3 classes specified using strings for categorical variable values
   - Loss function: sparse categorical cross-entropy
   - Metric: sparse categorical accuracy
   - Cross-Validation for generating training/validation/testing data sets

- [Wisconsin Diagnostic Breast Cancer](breast_cancer/README.md)
   - Problem type: classification
   - Input data: feature vector
   - Model type: fully-connected
   - Data file type: csv
   - 30 input features
   - Estimate the probability of being in class M
   - Loss function: binary cross-entropy
   - Metric: binary accuracy
   - Cross-Validation for generating training/validation/testing data sets

- [Daily Rainfall Retrieval](mesonet/README.md): Oklahoma Mesonet
rainfall estimation from other weather sensors
   - Problem type: both regression and classification
   - Input data: feature vector
   - Model type: fully-connected
   - Data file type: csv
   - ?? input features
   - Can be configured as either a continuous estimation problem or a rain/no rain classification problem
   - Loss functions: mean absolute error (mae) / Sparse categorical cross-entropy
   - Metrics: n/a / Sparse categorical accuracy
   - Cross-Validation for generating training/validation/testing data sets

- [Predicting Peptide Binding from Amino Acid
Sequences](amino/README.md)
   - Problem type: both regression and classification
   - Input data: strings of amino acid sequences
   - Model type: examples with fully-connected, convolutional neural network (1D), and recurrent neural network
      - Tokenization by character (one character per amino acid)
      - Embedding: translate integer token ID to a vector
   - Data file type: csv
   - One input feature: variable length string
   - One output: either scaled binding affinity (0 ... 1) or binary class
   - Loss functions: mean absolute error or binary cross-entropy
   - Metrics: mse/mae or binary accuracy
   - N-fold cross-validation
   - TF Datasets
      - Example of translating CSV file to TF Dataset folds
      - Use of TF Dataset folds instead of CSV file

- [Sentiment140 Natural Language Processing](sentiment140/README.md)
   - Problem type: classification
   - Input data: strings of natural language
   - Model type: recurrent neural network
      - Tokenization by whitespace (each token is a unique word)
      - Embedding: translate integer token ID to a vector
   - Data file type: csv
   - One input feature: string containing social media post
   - One output: whether the sentiment was positive (1) or negative (0)
   - Loss functions: binary cross-entropy
   - Metrics: binary accuracy
   - N-fold cross validation
   

- [Core 50 Image Recognition](core50/README.md): Object classification problem from images
   - Problem type: classification
   - Input data: images
   - Model type: convolutional neural network (2D)
   - Data file type: csv that refers to a set of images (tabular-indirect)
      - Note: the image data are not included in this repository
   - Images are 128x128 in size
   - Output: 2 classes (though there are 10 available in the full data set)
   - Loss function: sparse categorical cross-entropy
   - Metrics: sparse categorical accuracy
   - Cross-Validation


## Test Examples
In addition to the standard examples, we also have a set of tests that exercise different Zero2Neuro features.  See [../tests/index.md](Test Example Documentation)


