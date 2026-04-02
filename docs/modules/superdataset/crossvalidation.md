---
title: Cross-Validation
nav_order: 30
parent: Data Set Generator
has_children: true
---

### Holistic Cross-Validation
N-fold Cross-Validation is about creating N different models with
different subsets of data used as training, validation, and testing
data sets.  In this example, the data from the three files are
randomly split into 10 differnt groups (called _folds_).  Eight of these
folds are assigned to the training set, and one each is assigned to
the validation and testing sets.  The exact assigment is determined by
the _rotation_ argument.


```
--data_format=tabular
--data_split=random
--n_folds=10
--data_files
data_A.csv
data_B.csv
data_C.csv
--data_set_type=holistic-cross-validation
--data_inputs
In0
In1
--data_outputs
Out0
Out2
Out3
--rotation=3
```

