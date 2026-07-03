---
title: Scikit-Learn Preprocessors
nav_order: 10
parent: Scikit-Learn Models
has_children: false
---

# Scikit-Learn Preprocessors

The following Scikit-Learn module types are feature preprocessors.

## Modules

1. [MaxAbsScaler](#maxabsscaler)
2. [MinMaxScaler](#minmaxscaler)
3. [Normalizer](#normalizer)
4. [PolynomialFeatures](#polynomialfeatures)
5. [PowerTransformer](#powertransformer)
6. [QuantileTransformer](#quantiletransformer)
7. [RobustScaler](#robustscaler)
8. [SplineTransformer](#splinetransformer)
9. [StandardScaler](#standardscaler)

___

## MaxAbsScaler

Scikit-Learn module documentation: [MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html)

### Arguments

No Zero2Neuro arguments are currently mapped for this model; all parameters use Scikit-Learn defaults.

___

## MinMaxScaler

Scikit-Learn module documentation: [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

### Arguments

No Zero2Neuro arguments are currently mapped for this model; all parameters use Scikit-Learn defaults.

___

## Normalizer

Scikit-Learn module documentation: [Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Norm to use for normalization | `--skl_norm` | `l1` \| `l2` \| `max` | `None` | `norm` |

___

## PolynomialFeatures

Scikit-Learn module documentation: [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Degree of polynomial features | `--skl_poly_degree` | Integer ≥ 1 | Required | `degree` |
| Whether to produce only interaction features | `--skl_poly_interaction_only` | `True` \| `False` | Required | `interaction_only` |
| Whether to include a bias (constant) column | `--skl_include_bias` | `True` \| `False` | Required | `include_bias` |

___

## PowerTransformer

Scikit-Learn module documentation: [PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Power transform method | `--skl_method` | `yeo-johnson` \| `box-cox` | `None` | `method` |
| Whether to zero-mean, unit-variance the output | `--skl_standardize` | `True` \| `False` | `None` | `standardize` |

___

## QuantileTransformer

Scikit-Learn module documentation: [QuantileTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.QuantileTransformer.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Number of quantiles for transformation | `--skl_n_quantiles` | Integer ≥ 1 | `None` | `n_quantiles` |
| Target output distribution | `--skl_output_distribution` | `uniform` \| `normal` | `None` | `output_distribution` |

___

## RobustScaler

Scikit-Learn module documentation: [RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to center the data before scaling | `--skl_with_centering` | `True` \| `False` | `None` | `with_centering` |
| Whether to scale the data to unit variance | `--skl_with_scaling` | `True` \| `False` | `None` | `with_scaling` |

___

## SplineTransformer

Scikit-Learn module documentation: [SplineTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to include a bias (constant) column | `--skl_include_bias` | `True` \| `False` | `True` | `include_bias` |
| Number of knots | `--skl_n_knots` | Integer ≥ 2 | `None` | `n_knots` |
| Degree of the spline polynomial | `--skl_poly_degree` | Integer ≥ 0 | `None` | `degree` |

___

## StandardScaler

Scikit-Learn module documentation: [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

### Arguments

No Zero2Neuro arguments are currently mapped for this model; all parameters use Scikit-Learn defaults.

___
