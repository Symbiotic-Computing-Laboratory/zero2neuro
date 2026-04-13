---
title: Zero2Neuro Tests
nav_order: 35
has_children: true
parent: Zero2Neuro
---

# Zero2Neuro Tests

The following tests exercise specific Zero2Neuro features,
covering different data formats, data configurations, and
training options.

- [Categorical Variables](categorical_vars/README.md): Testing
the translation of categorical variables into integer representations
   - Problem type: XOR
   - Data file type: csv and pickle
   - Case 1: Boolean strings (T/F) translated to 1 and 0
   - Case 2: Integer-valued categorical columns (31/32 translated to 1 and 0)
   - Cases 3 & 4: Pickle versions of Cases 1 and 2

- [Tabular Test](tabular_test/README.md): Testing different header
row and data column configurations using the XOR problem
   - Problem type: XOR
   - Input data: binary feature vector
   - Model type: fully-connected (2 hidden layers: 20 and 10 units)
   - Data file type: csv
   - Multiple data configurations: 
      - Header row is non-standard (explicitly specified)
      - Case 1: Define a range of columns to use
      - Case 2: Define a list of columns to use

- [Tabular XLSX Test](tabular_xlsx_test/README.md): Testing different header row and data column configurations using xlsx format
   - Problem type: XOR
   - Input data: binary feature vector
   - Model type: fully-connected (2 hidden layers: 20 and 10 units)
   - Data file type: xlsx (multiple sheets)
   - Multiple data configurations: 
      - Case 1: single data table (maps to training set)
      - Case 2: two sheets in the same file are data tables 0 and 1, which are mapped to training and validation data sets, respectively

- [Data Conversion Test](data_conversion_test/README.md): Each data table is converted from a numpy array to a tf-dataset
   - Problem type: XOR
   - Input data: binary feature vector
   - Model type: fully-connected (2 hidden layers: 20 and 10 units)
   - Data file type: xlsx

- [Tabular Weights Test](tabular_weights_test/README.md): Testing
per-sample weights in tabular data using the XOR problem
   - Problem type: XOR
   - Input data: binary feature vector
   - Model type: fully-connected (2 hidden layers: 20 and 10 units)
   - Data file type: csv
   - Demonstrates the effect of different sample weights on model training
   - Look for: trained model will perform properly for the first three examples, but the output for the last example will typically be incorrect

- [Pickle Test](pickle_test/README.md): Testing the pickle data format with a parity problem
   - Data file type: pickle
   - Data representation: numpy
   - Multiple data configurations across 6 variants
   - Fixed train/validation/testing data set splits
