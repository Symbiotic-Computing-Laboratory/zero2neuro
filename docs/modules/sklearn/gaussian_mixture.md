---
title: Scikit-Learn Gaussian Mixture Models
nav_order: 80
parent: Scikit-Learn Models
has_children: false
---

# Scikit-Learn Gaussian Mixture Models

The following Scikit-Learn module types are Gaussian mixture models for probabilistic density estimation.

## Modules

1. [BayesianGaussianMixture](#bayesiangaussianmixture)
2. [GaussianMixture](#gaussianmixture)

___

## BayesianGaussianMixture

Scikit-Learn module documentation: [BayesianGaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of mixture components | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Type of covariance parameters | `--skl_covariance_type` | `full` \| `tied` \| `diag` \| `spherical` | `None` | `covariance_type` |
| Maximum number of EM iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |
| Number of initializations | `--skl_n_init` | Integer ≥ 1 | `None` | `n_init` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## GaussianMixture

Scikit-Learn module documentation: [GaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of mixture components | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Type of covariance parameters | `--skl_covariance_type` | `full` \| `tied` \| `diag` \| `spherical` | `None` | `covariance_type` |
| Maximum number of EM iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |
| Number of initializations | `--skl_n_init` | Integer ≥ 1 | `None` | `n_init` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___
