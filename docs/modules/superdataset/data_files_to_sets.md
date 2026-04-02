---
title: Data Files to Data Sets
nav_order: 30
parent: SuperDataSet
has_children: true
---

# Data Files to Data Sets

The translation from the contents of a set of files to the
training/validation/testing data sets is handled as a multi-step
process, with two intermediate representations.


<img src="../../../images/superdataset_detail.png" height=350>

The translation process is as follows:

1. __File Loader__: Each data file is loaded into a single _data
table_.  Each table consists of multiple data examples.
   - Supports a range of different file formats
   - Each example includes one or more inputs, one or more outputs,
(optional) example weight, and (optional) example group
   - [File Loader Details](./file_loader.md)

2. __Fold Generator__: the examples contained within the data tables
are sorted into one or more _data folds_.
   - The sorting process can be done table-by-table or example-by-example
   - [Fold Generator Details](./data_files_to_folds.md)

3. __Dataset Generator__: Assembles training/validation/testing data
sets by combining non-overlapping data folds.  Specifically:

   - The __training set__ is one or more folds

   - The __validation set__ is zero or more folds

   - The __testing set__ is zero or one fold

   - [Dataset Generator Details](./data_folds_to_sets.md)
