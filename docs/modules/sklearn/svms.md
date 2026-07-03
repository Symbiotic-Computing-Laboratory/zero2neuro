---
title: Scikit-Learn Support Vector Machines
nav_order: 110
parent: Scikit-Learn Models
has_children: false
---

# Scikit-Learn Support Vector Machines

The following Scikit-Learn module types are support vector machine models.

## Modules

1. [LinearSVC](#linearsvc)
2. [LinearSVR](#linearsvr)
3. [NuSVC](#nusvc)
4. [NuSVR](#nusvr)
5. [OneClassSVM](#oneclasssvm)
6. [SVC](#svc)
7. [SVR](#svr)

___

## LinearSVC

Scikit-Learn module documentation: [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit the intercept | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Regularization parameter | `--skl_C` | Float > 0 | `None` | `C` |
| Norm used in penalization | `--skl_penalty` | `l1` \| `l2` | `None` | `penalty` |
| Loss function | `--skl_loss` | `hinge` \| `squared_hinge` | `None` | `loss` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## LinearSVR

Scikit-Learn module documentation: [LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit the intercept | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Regularization parameter | `--skl_C` | Float > 0 | `None` | `C` |
| Loss function | `--skl_loss` | `epsilon_insensitive` \| `squared_epsilon_insensitive` | `None` | `loss` |
| Epsilon in the epsilon-insensitive loss tube | `--skl_epsilon` | Float ≥ 0 | `None` | `epsilon` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## NuSVC

Scikit-Learn module documentation: [NuSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Upper bound on training errors and support vectors | `--skl_nu` | Float in (0, 1] | `None` | `nu` |
| Kernel function | `--skl_kernel` | `linear` \| `poly` \| `rbf` \| `sigmoid` \| `precomputed` | `None` | `kernel` |
| Degree of polynomial kernel | `--skl_poly_degree` | Integer ≥ 1 | `None` | `degree` |
| Kernel coefficient | `--skl_gamma` | `scale` \| `auto` \| numeric string | `None` | `gamma` |
| Independent term in kernel function | `--skl_coef0` | Float | `None` | `coef0` |
| Maximum number of iterations (-1 for no limit) | `--skl_max_iter` | Integer ≥ 1 or -1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## NuSVR

Scikit-Learn module documentation: [NuSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Fraction of training errors and support vectors | `--skl_nu` | Float in (0, 1] | `None` | `nu` |
| Regularization parameter | `--skl_C` | Float > 0 | `None` | `C` |
| Kernel function | `--skl_kernel` | `linear` \| `poly` \| `rbf` \| `sigmoid` \| `precomputed` | `None` | `kernel` |
| Degree of polynomial kernel | `--skl_poly_degree` | Integer ≥ 1 | `None` | `degree` |
| Kernel coefficient | `--skl_gamma` | `scale` \| `auto` \| numeric string | `None` | `gamma` |
| Independent term in kernel function | `--skl_coef0` | Float | `None` | `coef0` |
| Maximum number of iterations (-1 for no limit) | `--skl_max_iter` | Integer ≥ 1 or -1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## OneClassSVM

Scikit-Learn module documentation: [OneClassSVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Upper bound on training errors | `--skl_nu` | Float in (0, 1] | `None` | `nu` |
| Kernel function | `--skl_kernel` | `linear` \| `poly` \| `rbf` \| `sigmoid` \| `precomputed` | `None` | `kernel` |
| Degree of polynomial kernel | `--skl_poly_degree` | Integer ≥ 1 | `None` | `degree` |
| Kernel coefficient | `--skl_gamma` | `scale` \| `auto` \| numeric string | `None` | `gamma` |
| Independent term in kernel function | `--skl_coef0` | Float | `None` | `coef0` |
| Maximum number of iterations (-1 for no limit) | `--skl_max_iter` | Integer ≥ 1 or -1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## SVC

Scikit-Learn module documentation: [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Regularization parameter | `--skl_C` | Float > 0 | `None` | `C` |
| Kernel function | `--skl_kernel` | `linear` \| `poly` \| `rbf` \| `sigmoid` \| `precomputed` | `None` | `kernel` |
| Degree of polynomial kernel | `--skl_poly_degree` | Integer ≥ 1 | `None` | `degree` |
| Kernel coefficient | `--skl_gamma` | `scale` \| `auto` \| numeric string | `None` | `gamma` |
| Independent term in kernel function | `--skl_coef0` | Float | `None` | `coef0` |
| Maximum number of iterations (-1 for no limit) | `--skl_max_iter` | Integer ≥ 1 or -1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## SVR

Scikit-Learn module documentation: [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Regularization parameter | `--skl_C` | Float > 0 | `None` | `C` |
| Kernel function | `--skl_kernel` | `linear` \| `poly` \| `rbf` \| `sigmoid` \| `precomputed` | `None` | `kernel` |
| Degree of polynomial kernel | `--skl_poly_degree` | Integer ≥ 1 | `None` | `degree` |
| Kernel coefficient | `--skl_gamma` | `scale` \| `auto` \| numeric string | `None` | `gamma` |
| Independent term in kernel function | `--skl_coef0` | Float | `None` | `coef0` |
| Epsilon in the epsilon-insensitive loss tube | `--skl_epsilon` | Float ≥ 0 | `None` | `epsilon` |
| Maximum number of iterations (-1 for no limit) | `--skl_max_iter` | Integer ≥ 1 or -1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___
