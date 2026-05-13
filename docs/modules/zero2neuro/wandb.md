---
title: Weights and Biases
nav_order: 30
parent: Zero2Neuro Engine
has_children: false
---

# Weights and Biases Support
_Weights and Biases_ is an online dashboard system that provides a real-time view of the status of all of your DNN experiments.  In particular, WandB provides:
- Visualization of all learning curves (loss and metrics as a function of training epoch for both the training and validation data sets).
- Visualization of the state of your computer as a function of time (CPU, GPU, and memory use).
- A picture of your model architecture.
- A full accounting of all of the argument values associated with each experiment.
- Post-training training, validation, and testing set performance.

## Getting Started (do once)

1. Create an account on the (Weights and Biases)[https://wandb.ai] web site.
2. Activate your python environment (e.g., ```conda activate dnn```)
3. Associate your local or supercomputer account with your WandB account
   - In your shell:
```
wandb login
```

4. Paste your API key when requested.  API key can be found in your W&B account settings.


More information: [detailed instructions](https://docs.wandb.ai/models/ref/cli/wandb-login)


## Zero2Neuro Arguments

You control the interaction of Zero2Neuro with WandB using the following command-line arguments:

- ```--wandb``` switch that turns on the use of Weights and Biases
- ```--wandb_project``` specifies the string project name.  WandB organizes your experiments at the high level in terms of _projects_.
- ```--wandb_name``` specifies the string name of your specific experiment.  These names index specific experiments for the specified project.  This string can reference any Zero2Neuro argument.  Default: "{args.experiment_name}", which makes use of the defined experiment name (which, in turn, can also reference Zero2Neuro arguments).
- ```--note``` specifies a string note that can be a useful text description of the specifics of your experiment.

