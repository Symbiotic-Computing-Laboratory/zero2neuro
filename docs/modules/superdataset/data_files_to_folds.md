---
title: Fold Generator
nav_order: 20
parent: Data Files to Data Sets
---

# Fold Generator: Data Tables to Data Folds

The Fold Generator is responsible for partitioning the full set of
examples into a group of folds, which are the atomic unit for
assembling training/validation/testing data sets.  Zero2Neuro provides
a range of options, depending on the organization of the incoming data
and the goals of the experiment.

The specific approach is determined by the argument:
```
--data_fold_split=YYY
```

where YYY is one of identity, group-by-file, group-by-example, and random

## Identity

The _identity_ split simply performs a one-to-one map of data tables
to folds.  
```
--data_fold_split=identity
--data_files
data_A.csv
data_B.csv
data_C.csv
```
For this example, data table A becomes fold 0, table B
becomes fold 1, and table C becomes fold 2.


## Group-by-File
Multiple data tables can be merge together for the purposes of
creating a single fold.  This mapping is explicitly defined in the
*data_files* declaration. 
```
--data_split=group-by-file
--data_files
data_A.csv,0
data_B.csv,0
data_C.csv,1
```

In this example, the data examples from 
data_A and data_B are merged into fold 0.  The examples from data_C
are assigned to fold 1.


### Random Split
When there is no explicit relationship between data files and the
examples that contain them, it is common to randomly split the
examples into folds.

```
--data_split=random
--n_folds=3
--data_files
data_A.csv
data_B.csv
data_C.csv
```

In this example, all of the data examples contained in data_A, data_B,
and data_C are randomly partitioned into 3 separate folds.

Notes:
- It is not appropriate to use this method if there is substantial
autocorrelation between data examples within individual files (e.g.,
if one file contains all of the data examples derived from a single
experimental subject).
- If one is performing a classification task and there is an imbalance
between the number of examples between the classes, then it is
appropriate to use the random-stratify split method.

## Random Split

## Group-by-Example

The tabular files can also contain an explicit assignment of
individual examples to folds.  In this situation, a column is added to
each of the tabular files; this column must be declared using the
*data_groups* argument:

```
--data_fold_split=group-by-example
--data_groups=FIELD
```
where FIELD is the column name.


- **FUTURE** **Group-by-Example**: the user explicitly assigns individual
examples to specific folds using _--data_folds_

- **Random**: the examples from all data tables are randomly assigned
to the folds.  
```
	--data_fold_split=random
	--n_folds=NNN
```
where NNN is the total number of folds.

- **FUTURE** **Random-Stratify**: the examples from all data tables
are randomly assigned to the folds.  This random assignment respects
the distribution of the __stratification field__
```
	--data_fold_split=random
	--n_folds=NNN
	--data_stratify=FIELD
```
where NNN is the total number of folds and FIELD is the field that
specifies the stratification class.

___

```
--data_fold_split=XXX
```

## Identity (default)

```
--data_fold_split=identity
```

## Group-by-File
```
--data_fold_split=group-by-file
```

## Group-by-Example
```
--data_fold_split=group-by-example
```

## Random
```
--data_fold_split=random
```

## Random with Stratification
```
--data_fold_split=random-stratify
```

