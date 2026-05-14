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

## Detailed Example
The [Full Example](./docs/examples/full_example.md) demonstrates the configuration and execution process for solving a simple logic problem.

## Dependencies
Zero2Neuro is implemented in Python and built on top of Keras3.  A full set of machine-readable dependencies is [available for download](requirements.txt)

## Documentation Links
- [Why Zero2Neuro](./docs/why_zero2neuro.md)
- [What's New](./docs/whats_new.md)
- [Full Example](./docs/examples/full_example.md)
- [Getting Started](./docs/getting_started/)
- [Zero2Neuro Examples](./examples/)
- [Zero2Neuro Tests](./tests/)
- [Core Modules](./docs/modules/)
- [API](./docs/api/)
- [Frequently Asked Questions](./docs/faq/)
- [Road Map](./docs/roadmap.md)
- [Contributors](./docs/contributors.md)

## References
- [Zero2Neuro Repository](https://github.com/Symbiotic-Computing-Laboratory/zero2neuro)
- [Zero2Neuro Presentation](https://docs.google.com/presentation/d/12ZBsMVq-6mW498PQZfDNP1_Mfu_3sIMWwnczJN84O5I/edit?usp=sharing)

