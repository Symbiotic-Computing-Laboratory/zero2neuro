---
title: Desired Output Preprocessing
nav_order: 120
parent: Scikit-Learn Models
has_children: false
---

# Desired Output Preprocessing
In some situations, the Zero2Neuro representation for desired outputs requires one or more steps of transformation before they are in a format that can be used with the corresponding Scikit-Learn model.  These transformations are implemented using a separate Pipeline that is applied just to the desired outputs.

## Example 

```
--skl_y_pipeline
TransformOutputRavel
```

## Modules

1. [TransformOutputRavel](#transformoutputravel)

___

## TransformOutputRavel
This module translates desired outputs of an arbitrary shape to a single vector.  Typically, this is used to translate a matrix of desired values of shape (N,1)  (N examples and one value for each example) to a vector that has a shape (N,) (still N examples)

### Arguments

There are no arguments for this preprocessing module.

___


