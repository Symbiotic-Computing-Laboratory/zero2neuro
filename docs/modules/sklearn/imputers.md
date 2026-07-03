---
title: Scikit-Learn Imputers
nav_order: 70
parent: Scikit-Learn Models
has_children: false
---

# Scikit-Learn Imputers

The following Scikit-Learn module types are missing value imputation methods.

## Modules

1. [KNNImputer](#knnimputer)
2. [MissingIndicator](#missingindicator)
3. [SimpleImputer](#simpleimputer)

___

## KNNImputer

Scikit-Learn module documentation: [KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of nearest neighbors for imputation | `--skl_n_neighbors` | Integer ≥ 1 | `None` | `n_neighbors` |
| Weight function for imputation | `--skl_weights` | `uniform` \| `distance` | `None` | `weights` |
| Distance metric | `--skl_metric` | String; see sklearn docs | `None` | `metric` |

___

## MissingIndicator

Scikit-Learn module documentation: [MissingIndicator](https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Which features to mark as missing | `--skl_features` | `missing-only` \| `all` | `None` | `features` |

___

## SimpleImputer

Scikit-Learn module documentation: [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Strategy for imputing missing values | `--skl_strategy` | `mean` \| `median` \| `most_frequent` \| `constant` | `None` | `strategy` |
| Fill value when using `constant` strategy | `--skl_fill_value` | String | `None` | `fill_value` |

___
