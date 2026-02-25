# Zero2Neuro Examples

- [Exclusive-Or](xor/README.md): simple 2-input, 1-output logic problem.  
   - Data file type: csv
   - Loss function: mean squared error

- [Iris](iris/README.md): iris flower classification problem
   - Data file type: csv
   - 4 input features
   - 3 classes specified using strings for categorical variable values
   - Loss function: sparse categorical cross-entropy
   - Metric: sparse categorical accuracy
   - Cross-Validation

- [Breast Cancer](breast_cancer/README.md)
   - Data file type: csv
   - 30 input features
   - 2 classes specified using strings for categorical variable values
   - Loss function: sparse categorical cross-entropy
   - Metric: sparse categorical accuracy
   - Cross-Validation

- [Daily Rainfall Retrieval](mesonet/README.md): Oklahoma Mesonet
rainfall estimation from other weather sensors
   - Data file type: csv
   - ?? input features
   - Can be configured as either a continuous estimation problem
or a rain/no rain classification problem
   - Loss functions: mean absolute error (mae) / Sparse
categorical cross-entropy
   - Metrics: n/a / Sparse categorical accuracy
   - Cross-Validation

- [Predicting Peptide Binding from Amino Acid Sequences](amino/README.md)
   - Data file type: csv
   - One input feature: variable length string
   - One output: either scaled binding affinity (0 ... 1) or binary class
   - Loss functions: mean absolute error or binary cross-entropy
   - Metrics: mse/mae or binary accuracy
   - N-fold cross-validation
   - Tokenization by character (one character per amino acid)
   - Embedding: translate integer token ID to a vector
   - TF Datasets
      - Example of translating CSV file to TF Dataset folds
      - Use of TF Dataset folds instead of CSV file

- [Core 50 Image Recognition](core50/README.md): Object classification
problem from images
   - Data file type: csv that refers to a set of images
   - Images are 128x128 in size
   - 2 classes (though there are 10 in the data set)
   - Loss function: sparse categorical cross-entropy
   - Metrics: sparse categorical accuracy
   - Cross-Validation


## Test Examples
In addition to the standard examples, we also have a set of tests that
exercise different Zero2Neuro features.


