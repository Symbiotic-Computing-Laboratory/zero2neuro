---
title: Zero2Neuro
nav_order: 1
has_children: true
layout: home
---
# Zero2Neuro 
<img SRC="images/zero2neuro2.png" height="300" alt="Zero2Neuro Icon" style="height:300px;">

Zero2Neuro is a no-code toolbox for constructing, training and
evaluating Deep Neural Network (DNN) models for a wide range of modeling
problems.  This package provides easy-to-use solutions for:

1. Loading data stored in a variety of formats (including the common
Comma Separated Values format), and configuring the data for use in
DNN training and evaluation.

2. Creation of Deep Neural Network models from several flexible DNN
model schemata, including fully-connected networks (FCNs), convolutional
neural networks (CNNs), and U-Nets.  The user specifies the
structural details of their specific model.

3. A standard DNN training and evaluation engine.  This engine
supports the production of result reports in various formats,
including hooks for [Weights and Biases](https://wandb.ai).

## Supported Environments

You can execute Zero2Neuro in several different environments,
depending on your needs:  

- Bash shell command line
- Supercomputer using SLURM
- Jupyter Notebooks

## Flexible Configuration

The user specifies the details behind their specific DNN experiment
using a set of simple configuration files:

1. Data configuration includes:
	- Location of data files and their type
	- Individual data features to be extracted from the files that
capture the model inputs and desired outputs
	- Translation of data types for categorical variables,
including one-hot encoding
	- Creation of training, validation, and testing data sets

2. Model configuration includes:
	- Model schema
	- Form of the model inputs
	- Number and sizes of model layers
	- Non-linearities for hidden and output layers

3. Training/evaluation engine configuration includes:
	- Learning parameters
	- Loss functions and metrics
	- Early-stopping parameters
	- How experiment results are reported

## Dependencies
Zero2Neuro is implemented in Python and built on top of Keras3

## Documentation Links
- [Full Example](./getting_started/full_example.md)
- [Getting Started](./getting_started/)
- [Examples](./examples/index.md)
- [Modules](./modules/index.md)
- [API](./api/index.html)

## References
- [Zero2Neuro Repository](https://github.com/Symbiotic-Computing-Laboratory/zero2neuro)
- [Zero2Neuro Presentation](https://docs.google.com/presentation/d/12ZBsMVq-6mW498PQZfDNP1_Mfu_3sIMWwnczJN84O5I/edit?usp=sharing)




## Quick Start

### 1. Clone the Repositories

Place the repository clones inside of a common directory:
- [Zero2Neuro](https://github.com/Symbiotic-Computing-Laboratory/zero2neuro)
- [Keras 3 Tools](https://github.com/Symbiotic-Computing-Laboratory/keras3_tools)

### 2. Declare Path
Execute:

```
export NEURO_REPOSITORY_PATH=/path/to/common/directory
```

### 3. Activate Python Environment

Activate Keras 3 / Tensorflow xx environment

Example:  
`conda activate tf`

### 4. Example: XOR

Change your directory:
```
cd examples/xor
```

Execute:  
```
python $NEURO_REPOSITORY_PATH/zero2neuro/src/zero2neuro.py @network.txt @data.txt @experiment.txt -vvv

```

- [Full XOR Description](examples/xor/README.md)


## Documentation

