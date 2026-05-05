---
title: Jupyter Environments
nav_order: 20
parent: Getting Started
---

# Existing Jupyter Lab Environments

OU has a Jupyter server for use by OU students and faculty. To utilize this server login with your OU account at [jupyter.lib.ou.edu](https://jupyter.lib.ou.edu).

Once inside the Jupyter enviroment, click the plus icon in the top left and open the terminal. Then to get Zero2Neuro working follow these steps.
  
1. cd ../..
2. cd data
3. cp -r zero2neuro ~/

Now to run an experiment, follow the format of the example notbooks except replace``` neuro_path = os.getenv("NEURO_REPOSITORY_PATH")``` with

```
os.environ["NEURO_REPOSITORY_PATH"] = "/home/jovyan/zero2neuro"  
neuro_path = os.getenv("NEURO_REPOSITORY_PATH")  
```

Note: The Jupyter Server can be slower than running locally or on the supercomputer and is not appropiate for larger jobs and should be used for small experiments or testing.
