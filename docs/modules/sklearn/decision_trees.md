---
title: Scikit-Learn Decision Trees
nav_order: 40
parent: Scikit-Learn Models
has_children: false
---

# Scikit-Learn Decision Trees

The following Scikit-Learn module types are decision tree models.

## Modules

1. [DecisionTreeClassifier](#decisiontreeclassifier)
2. [DecisionTreeRegressor](#decisiontreeregressor)
3. [ExtraTreeClassifier](#extratreeclassifier)
4. [ExtraTreeRegressor](#extratreeregressor)

___

## DecisionTreeClassifier

Scikit-Learn module documentation: [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Quality measure for splits | `--skl_criterion` | `gini` \| `entropy` \| `log_loss` | `None` | `criterion` |
| Maximum depth of the tree | `--skl_max_depth` | Integer ≥ 1 | `None` | `max_depth` |
| Minimum samples required to split a node | `--skl_min_samples_split` | Integer ≥ 2 | `None` | `min_samples_split` |
| Minimum samples required at a leaf node | `--skl_min_samples_leaf` | Integer ≥ 1 | `None` | `min_samples_leaf` |
| Number of features to consider for best split | `--skl_max_features` | `sqrt` \| `log2` | `None` | `max_features` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## DecisionTreeRegressor

Scikit-Learn module documentation: [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Quality measure for splits | `--skl_criterion` | `squared_error` \| `friedman_mse` | `None` | `criterion` |
| | | `absolute_error` \| `poisson` | | |
| Maximum depth of the tree | `--skl_max_depth` | Integer ≥ 1 | `None` | `max_depth` |
| Minimum samples required to split a node | `--skl_min_samples_split` | Integer ≥ 2 | `None` | `min_samples_split` |
| Minimum samples required at a leaf node | `--skl_min_samples_leaf` | Integer ≥ 1 | `None` | `min_samples_leaf` |
| Number of features to consider for best split | `--skl_max_features` | `sqrt` \| `log2` | `None` | `max_features` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## ExtraTreeClassifier

Scikit-Learn module documentation: [ExtraTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Quality measure for splits | `--skl_criterion` | `gini` \| `entropy` \| `log_loss` | `None` | `criterion` |
| Maximum depth of the tree | `--skl_max_depth` | Integer ≥ 1 | `None` | `max_depth` |
| Minimum samples required to split a node | `--skl_min_samples_split` | Integer ≥ 2 | `None` | `min_samples_split` |
| Minimum samples required at a leaf node | `--skl_min_samples_leaf` | Integer ≥ 1 | `None` | `min_samples_leaf` |
| Number of features to consider for best split | `--skl_max_features` | `sqrt` \| `log2` | `None` | `max_features` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## ExtraTreeRegressor

Scikit-Learn module documentation: [ExtraTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeRegressor.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Quality measure for splits | `--skl_criterion` | `squared_error` \| `friedman_mse` | `None` | `criterion` |
| | | `absolute_error` \| `poisson` | | |
| Maximum depth of the tree | `--skl_max_depth` | Integer ≥ 1 | `None` | `max_depth` |
| Minimum samples required to split a node | `--skl_min_samples_split` | Integer ≥ 2 | `None` | `min_samples_split` |
| Minimum samples required at a leaf node | `--skl_min_samples_leaf` | Integer ≥ 1 | `None` | `min_samples_leaf` |
| Number of features to consider for best split | `--skl_max_features` | `sqrt` \| `log2` | `None` | `max_features` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___
