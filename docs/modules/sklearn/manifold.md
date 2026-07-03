---
title: Scikit-Learn Manifold Methods
nav_order: 90
parent: Scikit-Learn Models
has_children: false
---

# Scikit-Learn Manifold Methods

The following Scikit-Learn module types are non-linear dimensionality reduction methods.

## Modules

1. [ClassicalMDS](#classicalmds)
2. [Isomap](#isomap)
3. [LocallyLinearEmbedding](#locallylinearembedding)
4. [MDS](#mds)
5. [SpectralEmbedding](#spectralembedding)
6. [TSNE](#tsne)

___

## ClassicalMDS

Scikit-Learn module documentation: [MDS](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html)

`ClassicalMDS` is implemented using sklearn's `MDS` with `metric=True` (Principal Coordinate Analysis / Classical Multidimensional Scaling).

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of dimensions for the embedding | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Number of initializations | `--skl_n_init` | Integer ≥ 1 | `None` | `n_init` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float ≥ 0 | `None` | `eps` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## Isomap

Scikit-Learn module documentation: [Isomap](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Number of nearest neighbors | `--skl_n_neighbors` | Integer ≥ 1 | `None` | `n_neighbors` |
| Number of dimensions for the embedding | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Distance metric | `--skl_metric` | String; see sklearn docs | `None` | `metric` |

___

## LocallyLinearEmbedding

Scikit-Learn module documentation: [LocallyLinearEmbedding](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Number of nearest neighbors | `--skl_n_neighbors` | Integer ≥ 1 | `None` | `n_neighbors` |
| Number of dimensions for the embedding | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float ≥ 0 | `None` | `tol` |
| LLE algorithm variant | `--skl_method` | `standard` \| `hessian` \| `modified` \| `ltsa` | `None` | `method` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## MDS

Scikit-Learn module documentation: [MDS](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Number of dimensions for the embedding | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Number of initializations | `--skl_n_init` | Integer ≥ 1 | `None` | `n_init` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float ≥ 0 | `None` | `eps` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## SpectralEmbedding

Scikit-Learn module documentation: [SpectralEmbedding](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Number of dimensions for the embedding | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___

## TSNE

Scikit-Learn module documentation: [TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Number of dimensions for the embedding | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Perplexity for nearest neighbor consideration | `--skl_perplexity` | Float > 0 | `None` | `perplexity` |
| Learning rate for optimization | `--skl_shrinkage` | Float > 0 | `None` | `learning_rate` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Random seed for reproducibility | `--skl_random_state` | Integer | `None` | `random_state` |

___
