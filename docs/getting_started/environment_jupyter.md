---
title: Jupyter Environments
nav_order: 20
parent: Getting Started
---

# Existing Jupyter Lab Environments

OU has a Jupyter server for use by OU students and faculty. To utilize this server login with your OU account at [jupyter.lib.ou.edu](https://jupyter.lib.ou.edu).

Once inside the Jupyter environment, click the plus icon in the top left and open the terminal. Then to get Zero2Neuro working follow these steps.  

1. Open up the terminal by pressing the blue + and clicking terminal
1. Find the path to "/data/zero2neuro", by default (from your root directory cd ~) this is "../../data/zero2neuro"
2. Leave the terminal and go back to the notebook for execution 
3. Replace ```neuro_path = os.getenv("NEURO_REPOSITORY_PATH")``` with ```os.environ["NEURO_REPOSITORY_PATH"] = "[path to /data/zero2neuro]"```
4. Run the experiment

An example is provided at [XOR Jupyter Example](../../examples/xor/xor-jupyterhub-notebook.ipynb)

Note: The Jupyter Server can be slower than running locally or on the supercomputer and is not appropriate for larger jobs and should be used for small experiments or testing.
