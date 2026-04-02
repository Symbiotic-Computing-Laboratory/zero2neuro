---
title: Internal Data Representation
nav_order: 40
parent: SuperDataSet
---

### Internal Representation

Internally, data tables, data folds, and training/validation/testing
data sets are represented internally in one of two ways:

1. __Numpy Arrays__ (default) are appropriate for small data sets that can be
contained entirely within the available RAM.  This representation
offers the most flexibility in how data are handled.

```
--data_representation=numpy
```

2. __Tensorflow Datasets__ represent data sets as data processing
pipelines.  These are most appropriate for large data sets, especially
those that are expensive to load from disk.  This option allows for
training of models to begin before all data are loaded from disk, and
allow caching of data to high-speed storage.

```
--data_representation=tf-dataset
```

