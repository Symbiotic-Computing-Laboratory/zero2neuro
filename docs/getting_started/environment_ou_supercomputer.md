---
title: Python Environment on the OU Supercomputer
nav_order: 1
has_children: true
parent: Getting Started
---

# Python Environment on the OU Supercomputer


The [OU Supercomputer](https://www.ou.edu/oscer) is a resource open to
researchers and students in Oklahoma.  For access to this system, see
[OSCER Getting Started](https://www.ou.edu/oscer/getting-started).

While you are free to set up your [own Python
environment](environment_self.md) and [copy of
Zero2Neuro](repository_configuration.md), you may use existing copies
of both.

## Python Environment
We maintain an up-to-date version of Python, Keras3, and Tensorflow.
To access this environment, type the following:

```
. /home/fagg/tf_setup.sh
conda activate dnn
```

## Zero2Neuro Configuration
For the release version of the repositories, type in your bash shell:
```
export NEURO_REPOSITORY_PATH=/home/fagg/zero2neuro
```

Alternatively, if you would like to use the development version of the
repositories, type the following:
```
export NEURO_REPOSITORY_PATH=/home/fagg/zero2neuro_devel
```

You may also add these lines to your own .bashrc file.

## Batch Files
Most work on the OU supercomputer is conducted using the SLURM job
control system, with your _batch.sh_ file containing the details of
your computational experiment.  The above bash commands may be
included in your _batch.sh_ file.

