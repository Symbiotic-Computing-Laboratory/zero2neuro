---
title: Scikit-Learn Naive Bayes
nav_order: 100
parent: Scikit-Learn Models
has_children: false
---

# Scikit-Learn Naive Bayes

The following Scikit-Learn module types are Naive Bayes classifiers based on Bayes' theorem.

## Modules

1. [BernoulliNB](#bernoullinb)
2. [CategoricalNB](#categoricalnb)
3. [ComplementNB](#complementnb)
4. [GaussianNB](#gaussiannb)
5. [MultinomialNB](#multinomialnb)

___

## BernoulliNB

Scikit-Learn module documentation: [BernoulliNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Laplace smoothing parameter | `--skl_alpha` | Float ≥ 0 | `None` | `alpha` |
| Whether to learn class prior probabilities | `--skl_fit_prior` | `True` \| `False` | `None` | `fit_prior` |

___

## CategoricalNB

Scikit-Learn module documentation: [CategoricalNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Laplace smoothing parameter | `--skl_alpha` | Float ≥ 0 | `None` | `alpha` |
| Whether to learn class prior probabilities | `--skl_fit_prior` | `True` \| `False` | `None` | `fit_prior` |

___

## ComplementNB

Scikit-Learn module documentation: [ComplementNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Laplace smoothing parameter | `--skl_alpha` | Float ≥ 0 | `None` | `alpha` |
| Whether to learn class prior probabilities | `--skl_fit_prior` | `True` \| `False` | `None` | `fit_prior` |

___

## GaussianNB

Scikit-Learn module documentation: [GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Portion of variance added for numerical stability | `--skl_var_smoothing` | Float > 0 | `None` | `var_smoothing` |

___

## MultinomialNB

Scikit-Learn module documentation: [MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Laplace smoothing parameter | `--skl_alpha` | Float ≥ 0 | `None` | `alpha` |
| Whether to learn class prior probabilities | `--skl_fit_prior` | `True` \| `False` | `None` | `fit_prior` |

___
