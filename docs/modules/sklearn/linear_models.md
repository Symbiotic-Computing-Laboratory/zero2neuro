---
title: Scikit-Learn Linear Models
nav_order: 20
parent: Scikit-Learn Models
has_children: false
---

# Scikit-Learn Linear Models

The following Scikit-Learn module types are all linear models

## Modules

1. [ARDRegression](#ardregression)
2. [BayesianRidge](#bayesianridge)
3. [ElasticNet](#elasticnet)
4. [GammaRegressor](#gammaregressor)
5. [HuberRegressor](#huberregressor)
6. [Lars](#lars)
7. [Lasso](#lasso)
8. [LassoLars](#lassolars)
9. [LassoLarsIC](#lassolarsic)
10. [LinearRegression](#linearregression)
11. [LogisticRegression](#logisticregression)
12. [MultiTaskElasticNet](#multitaskelasticnet)
13. [MultiTaskLasso](#multitasklasso)
14. [OrthogonalMatchingPursuit](#orthogonalmatchingpursuit)
15. [PassiveAggressiveRegressor](#passiveaggressiveregressor)
16. [Perceptron](#perceptron)
17. [PoissonRegressor](#poissonregressor)
18. [QuantileRegressor](#quantileregressor)
19. [RANSACRegressor](#ransacregressor)
20. [Ridge](#ridge)
21. [RidgeClassifier](#ridgeclassifier)
22. [SGDClassifier](#sgdclassifier)
23. [SGDRegressor](#sgdregressor)
24. [TheilSenRegressor](#theilsenregressor)
25. [TweedieRegressor](#tweedieregressor)

___

## ARDRegression

Scikit-Learn module documentation: [ARDRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Maximum number of solver iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## BayesianRidge

Scikit-Learn module documentation: [BayesianRidge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Maximum number of solver iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## ElasticNet

Scikit-Learn module documentation: [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| L1 regularization strength | `--L1_regularization` | Float ≥ 0 | Required | `alpha` |
| L1 to L2 mixing ratio (0 = pure L2, 1 = pure L1) | `--skl_l1_ratio` | Float in [0, 1] | Required | `l1_ratio` |
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Maximum number of solver iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## GammaRegressor

Scikit-Learn module documentation: [GammaRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| L2 regularization strength | `--L2_regularization` | Float ≥ 0 | `None` | `alpha` |
| Solver algorithm | `--skl_solver` | `lbfgs` \| `newton-cholesky` | `None` | `solver` |
| Maximum number of solver iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## HuberRegressor

Scikit-Learn module documentation: [HuberRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| L2 regularization strength | `--L2_regularization` | Float ≥ 0 | `None` | `alpha` |
| Threshold distinguishing inliers from outliers | `--skl_epsilon` | Float > 1.0 | `None` | `epsilon` |
| Maximum number of solver iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## Lars

Scikit-Learn module documentation: [Lars](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Number of CPU cores to use for computation | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |

___

## Lasso

Scikit-Learn module documentation: [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| L1 regularization strength | `--L1_regularization` | Float ≥ 0 | Required | `alpha` |
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Maximum number of solver iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## LassoLars

Scikit-Learn module documentation: [LassoLars](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| L1 regularization strength | `--L1_regularization` | Float ≥ 0 | Required | `alpha` |
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Maximum number of solver iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |

___

## LassoLarsIC

Scikit-Learn module documentation: [LassoLarsIC](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Model selection criterion | `--skl_criterion` | `aic` \| `bic` | `None` | `criterion` |
| Maximum number of solver iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |

___

## LinearRegression

Scikit-Learn module documentation: [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | Required | `fit_intercept` |
| Number of CPU cores to use for computation | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |

___

## LogisticRegression

Scikit-Learn module documentation: [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Number of CPU cores to use for computation | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Inverse regularization strength | `--skl_C` | Float > 0 | `None` | `C` |
| Regularization penalty type | `--skl_penalty` | `l1` \| `l2` \| `elasticnet` \| `None` | `None` | `penalty` |
| Solver algorithm | `--skl_solver` | `lbfgs` \| `liblinear` \| `newton-cg` \| `newton-cholesky` \| `sag` \| `saga` | `None` | `solver` |
| L1 to L2 mixing ratio for elasticnet penalty | `--skl_l1_ratio` | Float in [0, 1] | `None` | `l1_ratio` |
| Maximum number of solver iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## MultiTaskElasticNet

Scikit-Learn module documentation: [MultiTaskElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| L1 regularization strength | `--L1_regularization` | Float ≥ 0 | Required | `alpha` |
| L1 to L2 mixing ratio (0 = pure L2, 1 = pure L1) | `--skl_l1_ratio` | Float in [0, 1] | Required | `l1_ratio` |
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Maximum number of solver iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## MultiTaskLasso

Scikit-Learn module documentation: [MultiTaskLasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| L1 regularization strength | `--L1_regularization` | Float ≥ 0 | Required | `alpha` |
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Maximum number of solver iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## OrthogonalMatchingPursuit

Scikit-Learn module documentation: [OrthogonalMatchingPursuit](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Target number of non-zero coefficients | `--skl_n_nonzero_coefs` | Integer ≥ 1 | `None` | `n_nonzero_coefs` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## PassiveAggressiveRegressor

Scikit-Learn module documentation: [PassiveAggressiveRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Inverse regularization strength | `--skl_C` | Float > 0 | `None` | `C` |
| Maximum number of training passes | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## Perceptron

Scikit-Learn module documentation: [Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Number of CPU cores to use for computation | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Initial learning rate | `--skl_eta0` | Float > 0 | `None` | `eta0` |
| Regularization penalty type | `--skl_penalty` | `l1` \| `l2` \| `elasticnet` | `None` | `penalty` |
| Maximum number of training passes | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## PoissonRegressor

Scikit-Learn module documentation: [PoissonRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| L2 regularization strength | `--L2_regularization` | Float ≥ 0 | `None` | `alpha` |
| Solver algorithm | `--skl_solver` | `lbfgs` \| `newton-cholesky` | `None` | `solver` |
| Maximum number of solver iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## QuantileRegressor

Scikit-Learn module documentation: [QuantileRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Quantile to estimate | `--skl_quantile` | Float in [0, 1] | `None` | `quantile` |
| L1 regularization strength | `--L1_regularization` | Float ≥ 0 | `None` | `alpha` |
| Solver algorithm | `--skl_solver` | `highs` \| `highs-ds` \| `highs-ipm` \| `interior-point` \| `revised simplex` | `None` | `solver` |

___

## RANSACRegressor

Scikit-Learn module documentation: [RANSACRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html)

### Arguments

No Zero2Neuro arguments are currently mapped for this model; all parameters use Scikit-Learn defaults.

___

## Ridge

Scikit-Learn module documentation: [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| L2 regularization strength | `--L2_regularization` | Float ≥ 0 | Required | `alpha` |
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Solver algorithm | `--skl_solver` | `auto` \| `svd` \| `cholesky` \| `lsqr` \| `sparse_cg` \| `sag` \| `saga` \| `lbfgs` | `None` | `solver` |
| Maximum number of solver iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |

___

## RidgeClassifier

Scikit-Learn module documentation: [RidgeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| L2 regularization strength | `--L2_regularization` | Float ≥ 0 | `None` | `alpha` |
| Solver algorithm | `--skl_solver` | `auto` \| `svd` \| `cholesky` \| `lsqr` \| `sparse_cg` \| `sag` \| `saga` \| `lbfgs` | `None` | `solver` |
| Maximum number of solver iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## SGDClassifier

Scikit-Learn module documentation: [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Number of CPU cores to use for computation | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Loss function | `--skl_loss` | `hinge` \| `log_loss` \| `modified_huber` \| `squared_hinge` \| `perceptron` | `None` | `loss` |
| | | `squared_error` \| `huber` \| `epsilon_insensitive` \| `squared_epsilon_insensitive` | | |
| Regularization penalty type | `--skl_penalty` | `l1` \| `l2` \| `elasticnet` | `None` | `penalty` |
| L2 regularization strength | `--L2_regularization` | Float ≥ 0 | `None` | `alpha` |
| L1 to L2 mixing ratio for elasticnet penalty | `--skl_l1_ratio` | Float in [0, 1] | `None` | `l1_ratio` |
| Initial learning rate | `--skl_eta0` | Float > 0 | `None` | `eta0` |
| Learning rate schedule | `--skl_learning_rate` | `constant` \| `optimal` \| `invscaling` \| `adaptive` | `None` | `learning_rate` |
| Maximum number of training passes | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## SGDRegressor

Scikit-Learn module documentation: [SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Loss function | `--skl_loss` | `squared_error` \| `huber` \| `epsilon_insensitive` \| `squared_epsilon_insensitive` | `None` | `loss` |
| Regularization penalty type | `--skl_penalty` | `l1` \| `l2` \| `elasticnet` | `None` | `penalty` |
| L2 regularization strength | `--L2_regularization` | Float ≥ 0 | `None` | `alpha` |
| L1 to L2 mixing ratio for elasticnet penalty | `--skl_l1_ratio` | Float in [0, 1] | `None` | `l1_ratio` |
| Initial learning rate | `--skl_eta0` | Float > 0 | `None` | `eta0` |
| Learning rate schedule | `--skl_learning_rate` | `constant` \| `optimal` \| `invscaling` \| `adaptive` | `None` | `learning_rate` |
| Maximum number of training passes | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## TheilSenRegressor

Scikit-Learn module documentation: [TheilSenRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| Number of CPU cores to use for computation | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Maximum number of solver iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___

## TweedieRegressor

Scikit-Learn module documentation: [TweedieRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Whether to fit a bias/intercept term | `--skl_include_bias` | `True` \| `False` | `True` | `fit_intercept` |
| L2 regularization strength | `--L2_regularization` | Float ≥ 0 | `None` | `alpha` |
| Tweedie distribution power parameter | `--skl_tweedie_power` | Float | `None` | `power` |
| Solver algorithm | `--skl_solver` | `lbfgs` \| `newton-cholesky` | `None` | `solver` |
| Maximum number of solver iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float > 0 | `None` | `tol` |

___
