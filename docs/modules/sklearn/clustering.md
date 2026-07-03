---
title: Scikit-Learn Clustering
nav_order: 30
parent: Scikit-Learn Models
has_children: false
---

# Scikit-Learn Clustering

The following Scikit-Learn module types are clustering methods.

## Modules

1. [AffinityPropagation](#affinitypropagation)
2. [AgglomerativeClustering](#agglomerativeclustering)
3. [Birch](#birch)
4. [BisectingKMeans](#bisectingkmeans)
5. [DBSCAN](#dbscan)
6. [FeatureAgglomeration](#featureagglomeration)
7. [HDBSCAN](#hdbscan)
8. [KMeans](#kmeans)
9. [MeanShift](#meanshift)
10. [MiniBatchKMeans](#minibatchkmeans)
11. [SpectralBiclustering](#spectralbiclustering)
12. [SpectralClustering](#spectralclustering)
13. [SpectralCoclustering](#spectralcoclustering)

___

## AffinityPropagation

Scikit-Learn module documentation: [AffinityPropagation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Damping factor for message-passing | `--skl_damping` | Float in [0.5, 1.0) | `None` | `damping` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |

___

## AgglomerativeClustering

Scikit-Learn module documentation: [AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of clusters to find | `--skl_n_clusters` | Integer ≥ 1 | `None` | `n_clusters` |
| Distance metric for linkage | `--skl_metric` | String; see sklearn docs | `None` | `metric` |
| Linkage criterion | `--skl_linkage` | `ward` \| `complete` \| `average` \| `single` | `None` | `linkage` |

___

## Birch

Scikit-Learn module documentation: [Birch](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of clusters for the final step | `--skl_n_clusters` | Integer ≥ 1 | `None` | `n_clusters` |
| Radius threshold for subclusters | `--skl_threshold` | Float > 0 | `None` | `threshold` |
| Maximum number of subclusters per node | `--skl_branching_factor` | Integer ≥ 1 | `None` | `branching_factor` |

___

## BisectingKMeans

Scikit-Learn module documentation: [BisectingKMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.BisectingKMeans.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of clusters to find | `--skl_n_clusters` | Integer ≥ 1 | `None` | `n_clusters` |
| Maximum number of iterations per bisection | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float ≥ 0 | `None` | `tol` |

___

## DBSCAN

Scikit-Learn module documentation: [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Maximum distance between neighborhood samples | `--skl_eps` | Float > 0 | `None` | `eps` |
| Minimum samples in a neighborhood | `--skl_min_samples` | Integer ≥ 1 | `None` | `min_samples` |
| Distance metric | `--skl_metric` | String; see sklearn docs | `None` | `metric` |

___

## FeatureAgglomeration

Scikit-Learn module documentation: [FeatureAgglomeration](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of clusters to find | `--skl_n_clusters` | Integer ≥ 1 | `None` | `n_clusters` |
| Distance metric for linkage | `--skl_metric` | String; see sklearn docs | `None` | `metric` |
| Linkage criterion | `--skl_linkage` | `ward` \| `complete` \| `average` \| `single` | `None` | `linkage` |

___

## HDBSCAN

Scikit-Learn module documentation: [HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Minimum number of samples in a cluster | `--skl_min_cluster_size` | Integer ≥ 2 | `None` | `min_cluster_size` |
| Minimum samples in a neighborhood | `--skl_min_samples` | Integer ≥ 1 | `None` | `min_samples` |
| Distance metric | `--skl_metric` | String; see sklearn docs | `None` | `metric` |

___

## KMeans

Scikit-Learn module documentation: [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of clusters to find | `--skl_n_clusters` | Integer ≥ 1 | `None` | `n_clusters` |
| Number of centroid initializations | `--skl_n_init` | Integer ≥ 1 | `None` | `n_init` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Convergence tolerance | `--skl_tol` | Float ≥ 0 | `None` | `tol` |

___

## MeanShift

Scikit-Learn module documentation: [MeanShift](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of CPU cores to use | `--cpus_per_task` | Integer ≥ 1 | `None` | `n_jobs` |
| Bandwidth for kernel density estimation | `--skl_bandwidth` | Float > 0 | `None` | `bandwidth` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |

___

## MiniBatchKMeans

Scikit-Learn module documentation: [MiniBatchKMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of clusters to find | `--skl_n_clusters` | Integer ≥ 1 | `None` | `n_clusters` |
| Number of centroid initializations | `--skl_n_init` | Integer ≥ 1 | `None` | `n_init` |
| Maximum number of iterations | `--skl_max_iter` | Integer ≥ 1 | `None` | `max_iter` |
| Number of samples per mini-batch | `--skl_batch_size` | Integer ≥ 1 | `None` | `batch_size` |
| Convergence tolerance | `--skl_tol` | Float ≥ 0 | `None` | `tol` |

___

## SpectralBiclustering

Scikit-Learn module documentation: [SpectralBiclustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralBiclustering.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of biclusters to find | `--skl_n_clusters` | Integer ≥ 1 | `None` | `n_clusters` |
| Number of singular vectors to use | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Number of k-means initializations | `--skl_n_init` | Integer ≥ 1 | `None` | `n_init` |

___

## SpectralClustering

Scikit-Learn module documentation: [SpectralClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of clusters to find | `--skl_n_clusters` | Integer ≥ 1 | `None` | `n_clusters` |
| Number of eigenvectors to use | `--skl_n_components` | Integer ≥ 1 | `None` | `n_components` |
| Number of k-means initializations | `--skl_n_init` | Integer ≥ 1 | `None` | `n_init` |

___

## SpectralCoclustering

Scikit-Learn module documentation: [SpectralCoclustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralCoclustering.html)

### Arguments

| Description | Zero2Neuro Argument | Possible Values | Default Value | Scikit-Learn Parameter |
|---|---|---|---|---|
| Number of biclusters to find | `--skl_n_clusters` | Integer ≥ 1 | `None` | `n_clusters` |
| Number of k-means initializations | `--skl_n_init` | Integer ≥ 1 | `None` | `n_init` |

___
