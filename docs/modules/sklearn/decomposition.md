---
title: Scikit-Learn Decomposition Methods
nav_order: 50
parent: Scikit-Learn Models
has_children: false
---

# Scikit-Learn Decomposition Methods

The following Scikit-Learn module types are matrix factorization and dimensionality reduction methods.

## Modules

1. [FactorAnalysis](#factoranalysis)
2. [FastICA](#fastica)
3. [IncrementalPCA](#incrementalpca)
4. [KernelPCA](#kernelpca)
5. [LatentDirichletAllocation](#latentdirichletallocation)
6. [MiniBatchNMF](#minibatchnmf)
7. [MiniBatchSparsePCA](#minibatchsparsepca)
8. [NMF](#nmf)
9. [PCA](#pca)
10. [SparseCoder](#sparsecoder)
11. [SparsePCA](#sparsepca)
12. [TruncatedSVD](#truncatedsvd)

___

## FactorAnalysis

Scikit-Learn module documentation: [FactorAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of latent components | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float ≥ 0 | `None` | `tol` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## FastICA

Scikit-Learn module documentation: [FastICA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of independent components | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float ≥ 0 | `None` | `tol` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## IncrementalPCA

Scikit-Learn module documentation: [IncrementalPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of principal components | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Number of samples per mini-batch | `--skl_batch_size` | Integer ≥ 1 | `None` | `batch_size` |

___

## KernelPCA

Scikit-Learn module documentation: [KernelPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Number of principal components | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Kernel function | `--skl_kernel` | `linear` \| `poly` \| `rbf` \| `sigmoid` \| `cosine` \| `precomputed` | `None` | `kernel` |
| Kernel coefficient | `--skl_gamma` | `scale` \| `auto` \| numeric string | `None` | `gamma` |
| Degree of polynomial kernel | `--skl_poly_degree` | Integer ≥ 1 | `None` | `degree` |
| Independent term in kernel function | `--skl_coef0` | Float | `None` | `coef0` |
| Maximum number of iterations for eigenvalue solver | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## LatentDirichletAllocation

Scikit-Learn module documentation: [LatentDirichletAllocation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Number of topics | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Number of documents per mini-batch | `--skl_batch_size` | Integer ≥ 1 | `None` | `batch_size` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## MiniBatchNMF

Scikit-Learn module documentation: [MiniBatchNMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchNMF.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of components | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float ≥ 0 | `None` | `tol` |
| Number of samples per mini-batch | `--skl_batch_size` | Integer ≥ 1 | `None` | `batch_size` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## MiniBatchSparsePCA

Scikit-Learn module documentation: [MiniBatchSparsePCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchSparsePCA.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Number of sparse components | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Number of samples per mini-batch | `--skl_batch_size` | Integer ≥ 1 | `None` | `batch_size` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## NMF

Scikit-Learn module documentation: [NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of components | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float ≥ 0 | `None` | `tol` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## PCA

Scikit-Learn module documentation: [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of principal components | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## SparseCoder

Scikit-Learn module documentation: [SparseCoder](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparseCoder.html)

**Note:** SparseCoder requires a `dictionary` array that must be provided externally and cannot be configured via CLI arguments.

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |

___

## SparsePCA

Scikit-Learn module documentation: [SparsePCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Number of sparse components | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float ≥ 0 | `None` | `tol` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## TruncatedSVD

Scikit-Learn module documentation: [TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of singular values to compute | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___
