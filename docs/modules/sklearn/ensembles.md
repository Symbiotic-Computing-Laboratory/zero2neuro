---
title: Scikit-Learn Ensemble Methods
nav_order: 60
parent: Scikit-Learn Models
has_children: false
---

# Scikit-Learn Ensemble Methods

The following Scikit-Learn module types are ensemble methods.

## Modules

1. [AdaBoostClassifier](#adaboostclassifier)
2. [AdaBoostRegressor](#adaboostregressor)
3. [BaggingClassifier](#baggingclassifier)
4. [BaggingRegressor](#baggingregressor)
5. [ExtraTreesClassifier](#extratreesclassifier)
6. [ExtraTreesRegressor](#extratreesregressor)
7. [GradientBoostingClassifier](#gradientboostingclassifier)
8. [GradientBoostingRegressor](#gradientboostingregressor)
9. [HistGradientBoostingClassifier](#histgradientboostingclassifier)
10. [HistGradientBoostingRegressor](#histgradientboostingregressor)
11. [IsolationForest](#isolationforest)
12. [RandomForestClassifier](#randomforestclassifier)
13. [RandomForestRegressor](#randomforestregressor)
14. [RandomTreesEmbedding](#randomtreesembedding)

___

## AdaBoostClassifier

Scikit-Learn module documentation: [AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of estimators to train | `--skl_n_estimators` | Integer ≥ 1 | `None` | `n_estimators` |
| Learning rate for boosting | `--skl_shrinkage` | Float > 0 | `None` | `learning_rate` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## AdaBoostRegressor

Scikit-Learn module documentation: [AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of estimators to train | `--skl_n_estimators` | Integer ≥ 1 | `None` | `n_estimators` |
| Learning rate for boosting | `--skl_shrinkage` | Float > 0 | `None` | `learning_rate` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## BaggingClassifier

Scikit-Learn module documentation: [BaggingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Number of estimators to train | `--skl_n_estimators` | Integer ≥ 1 | `None` | `n_estimators` |
| Fraction of samples to draw per estimator | `--skl_max_samples` | Float in (0, 1] | `None` | `max_samples` |
| Number of features to consider for each estimator | `--skl_max_features` | `sqrt` \| `log2` | `None` | `max_features` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## BaggingRegressor

Scikit-Learn module documentation: [BaggingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Number of estimators to train | `--skl_n_estimators` | Integer ≥ 1 | `None` | `n_estimators` |
| Fraction of samples to draw per estimator | `--skl_max_samples` | Float in (0, 1] | `None` | `max_samples` |
| Number of features to consider for each estimator | `--skl_max_features` | `sqrt` \| `log2` | `None` | `max_features` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## ExtraTreesClassifier

Scikit-Learn module documentation: [ExtraTreesClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Number of trees in the forest | `--skl_n_estimators` | Integer ≥ 1 | `None` | `n_estimators` |
| Quality measure for splits | `--skl_criterion` | `gini` \| `entropy` \| `log_loss` | `None` | `criterion` |
| Maximum depth of each tree | `--skl_max_depth` | Integer ≥ 1 | `None` | `max_depth` |
| Minimum samples required to split a node | `--skl_min_samples_split` | Integer ≥ 2 | `None` | `min_samples_split` |
| Minimum samples required at a leaf node | `--skl_min_samples_leaf` | Integer ≥ 1 | `None` | `min_samples_leaf` |
| Number of features to consider for best split | `--skl_max_features` | `sqrt` \| `log2` | `None` | `max_features` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## ExtraTreesRegressor

Scikit-Learn module documentation: [ExtraTreesRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Number of trees in the forest | `--skl_n_estimators` | Integer ≥ 1 | `None` | `n_estimators` |
| Quality measure for splits | `--skl_criterion` | `squared_error` \| `friedman_mse` | `None` | `criterion` |
| | | `absolute_error` \| `poisson` | | |
| Maximum depth of each tree | `--skl_max_depth` | Integer ≥ 1 | `None` | `max_depth` |
| Minimum samples required to split a node | `--skl_min_samples_split` | Integer ≥ 2 | `None` | `min_samples_split` |
| Minimum samples required at a leaf node | `--skl_min_samples_leaf` | Integer ≥ 1 | `None` | `min_samples_leaf` |
| Number of features to consider for best split | `--skl_max_features` | `sqrt` \| `log2` | `None` | `max_features` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## GradientBoostingClassifier

Scikit-Learn module documentation: [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of boosting stages | `--skl_n_estimators` | Integer ≥ 1 | `None` | `n_estimators` |
| Learning rate (shrinks each tree's contribution) | `--skl_shrinkage` | Float > 0 | `None` | `learning_rate` |
| Maximum depth of each tree | `--skl_max_depth` | Integer ≥ 1 | `None` | `max_depth` |
| Fraction of samples used per stage | `--skl_subsample` | Float in (0, 1] | `None` | `subsample` |
| Quality measure for splits | `--skl_criterion` | `friedman_mse` \| `squared_error` | `None` | `criterion` |
| Number of features to consider for best split | `--skl_max_features` | `sqrt` \| `log2` | `None` | `max_features` |
| Convergence tolerance | `--skl_tol` | Float ≥ 0 | `None` | `tol` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## GradientBoostingRegressor

Scikit-Learn module documentation: [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of boosting stages | `--skl_n_estimators` | Integer ≥ 1 | `None` | `n_estimators` |
| Learning rate (shrinks each tree's contribution) | `--skl_shrinkage` | Float > 0 | `None` | `learning_rate` |
| Maximum depth of each tree | `--skl_max_depth` | Integer ≥ 1 | `None` | `max_depth` |
| Fraction of samples used per stage | `--skl_subsample` | Float in (0, 1] | `None` | `subsample` |
| Quality measure for splits | `--skl_criterion` | `friedman_mse` \| `squared_error` | `None` | `criterion` |
| Number of features to consider for best split | `--skl_max_features` | `sqrt` \| `log2` | `None` | `max_features` |
| Convergence tolerance | `--skl_tol` | Float ≥ 0 | `None` | `tol` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## HistGradientBoostingClassifier

Scikit-Learn module documentation: [HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Learning rate (shrinks each tree's contribution) | `--skl_shrinkage` | Float > 0 | `None` | `learning_rate` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Maximum depth of each tree | `--skl_max_depth` | Integer ≥ 1 | `None` | `max_depth` |
| Convergence tolerance | `--skl_tol` | Float ≥ 0 | `None` | `tol` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## HistGradientBoostingRegressor

Scikit-Learn module documentation: [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Learning rate (shrinks each tree's contribution) | `--skl_shrinkage` | Float > 0 | `None` | `learning_rate` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Maximum depth of each tree | `--skl_max_depth` | Integer ≥ 1 | `None` | `max_depth` |
| Convergence tolerance | `--skl_tol` | Float ≥ 0 | `None` | `tol` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## IsolationForest

Scikit-Learn module documentation: [IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Number of trees in the ensemble | `--skl_n_estimators` | Integer ≥ 1 | `None` | `n_estimators` |
| Fraction of samples to draw per tree | `--skl_max_samples` | Float in (0, 1] | `None` | `max_samples` |
| Proportion of outliers in the dataset | `--skl_contamination` | Float in (0, 0.5] | `None` | `contamination` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## RandomForestClassifier

Scikit-Learn module documentation: [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Number of trees in the forest | `--skl_n_estimators` | Integer ≥ 1 | `None` | `n_estimators` |
| Quality measure for splits | `--skl_criterion` | `gini` \| `entropy` \| `log_loss` | `None` | `criterion` |
| Maximum depth of each tree | `--skl_max_depth` | Integer ≥ 1 | `None` | `max_depth` |
| Minimum samples required to split a node | `--skl_min_samples_split` | Integer ≥ 2 | `None` | `min_samples_split` |
| Minimum samples required at a leaf node | `--skl_min_samples_leaf` | Integer ≥ 1 | `None` | `min_samples_leaf` |
| Number of features to consider for best split | `--skl_max_features` | `sqrt` \| `log2` | `None` | `max_features` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## RandomForestRegressor

Scikit-Learn module documentation: [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Number of trees in the forest | `--skl_n_estimators` | Integer ≥ 1 | `None` | `n_estimators` |
| Quality measure for splits | `--skl_criterion` | `squared_error` \| `friedman_mse` | `None` | `criterion` |
| | | `absolute_error` \| `poisson` | | |
| Maximum depth of each tree | `--skl_max_depth` | Integer ≥ 1 | `None` | `max_depth` |
| Minimum samples required to split a node | `--skl_min_samples_split` | Integer ≥ 2 | `None` | `min_samples_split` |
| Minimum samples required at a leaf node | `--skl_min_samples_leaf` | Integer ≥ 1 | `None` | `min_samples_leaf` |
| Number of features to consider for best split | `--skl_max_features` | `sqrt` \| `log2` | `None` | `max_features` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## RandomTreesEmbedding

Scikit-Learn module documentation: [RandomTreesEmbedding](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomTreesEmbedding.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Number of trees in the forest | `--skl_n_estimators` | Integer ≥ 1 | `None` | `n_estimators` |
| Maximum depth of each tree | `--skl_max_depth` | Integer ≥ 1 | `None` | `max_depth` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___
