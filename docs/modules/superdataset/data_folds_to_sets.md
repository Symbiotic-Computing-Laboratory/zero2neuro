---
title: Data Set Generator
nav_order: 30
parent: Data Files to Data Sets
has_children: true
---

# Data Set Generator

- [Advanced: N-Fold Cross-Validation](./cross_validation.md)

## Folds to Data Sets

Folds are assembled into training, validation, and testing data sets
in one of several ways:

- **Fixed** (default)
	- Fold 0 is assigned to the training data set
	- Fold 1 (if it exists) is assigned to the validation data set
	- Fold 2 (if it exists) is assigned to the testing data set
	- An error will be raised if there are more than three folds

```
	--data_set_type=fixed
```

- **Holistic-Cross-Validation**: for N folds, N-2 folds are merged to
form the training set

```
	--data_set_type=holistic-cross-validation
	--rotation=RRR
```
where RRR is the rotation number (0 ... N-1).  Typically, this
argument is specified at the command line and not in a configuration
file. 

- **Hold-Out-Cross-Validation**

- **FUTURE** **Orthogonalized-Cross-Validation**

