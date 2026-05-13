# Weights & Biases Integration

The Zero2Neuro Engine supports optional usage of Weights & Biases (W&B) for experiment tracking and metric visualization.

When enabled Zero2Neuro automatically creates a W&B run and logs:

- training/validation/testing metrics
- resource usage
- model architecture visualizations

---

# Installation
If the packages for Zero2Neuro were installed using the (requirements.txt)[../../../requirements.txt) wandb is already included.

Otherwise install by using pip
```
pip install wandb
```

---
# Creating an account

1. Create account at [wandb.ai](https://wandb.ai)

2. Log into W&B in terminal
```
wandb login
```
3. Paste your API key when asked

API key can be found in your W&B account settings.

---

# Using W&B
W&B is enabled via the `--wandb` flag.

To specify the project name do `--wandb_project project_name` else it will default to Supernetwork.  
  
Zero2Neuro has automatic experiment names {args.experiment_name} but if you want to force it to something specific do `--wandb_name experiment_name`.

To attach a note to your run do `--note "Example note"` by default this is disabled but is helpful to keep track of changes you've made between runs.