[Base Index](index.md)    
# Getting Started

## Assumptions
- TODO
- You have git installed
- You have a python enviroment set up with Keras3 and TensorFlow
- You have a basic conceptual understanding of DNNs  
    - TODO: Add resources  

## Instructions  
**1. Clone the Repositories**

Place the repository clones inside of a common directory:
- [Zero2Neuro](https://github.com/Symbiotic-Computing-Laboratory/zero2neuro)
- [Keras 3 Tools](https://github.com/Symbiotic-Computing-Laboratory/keras3_tools)

**2. Declare Path**
Add to your .bashrc file:

`export NEURO_REPOSITORY_PATH=/path/to/common/directory`

**3. Python Environment**
Activate Keras 3 / Tensorflow xx environment

Example:  
`conda activate tf`

**4. Example: XOR**

Change your directory to: `examples/xor/`

Execute the batch script:  
`./batch.sh`

**5. Look at the Other Examples**

There are several other examples in `examples/` look at their README file for more information.

**6. Try your hand at your own model**

By now you should be ready to go with Zero2Neuro.  
  
Find a dataset online or input your own and set up a model using the examples as references.

## Setting up Python Enviroment  

1. Install Python and Python libraries  
[Python 3.12](https://www.python.org/downloads/release/python-3120/)  
[TensorFlow 2.16](https://www.tensorflow.org/install)  
[Keras 3.8](https://keras.io/getting_started/)  
[Scikit-learn 1.6](https://scikit-learn.org/stable/install.html)  
[Pandas 2.2](https://pandas.pydata.org/docs/dev/getting_started/install.html#install)  
  
3. Install either JupyterLab or other software that can read jupyter notebooks.  
[JupyterLab](https://jupyter.org/install)  
  
5. Declare Path Add to your .bashrc file (built in on unix-based systems and software):  
export NEURO_REPOSITORY_PATH=/path/to/common/directory  
  
## Supercomputer
  

Zero2Neuro was built with supercomputer usage in mind. For information about OU's supercomputer Schooner and how to utilize it visit https://www.ou.edu/oscer. Examples provided in [/examples](../examples/) contain `batch.sh` files to use as examples for running jobs on Schooner.  

## Support Information
If you have any questions or require further assistance contact the appropriate resource.  
  
For questions regarding **Zero2Neuro**: Contact Luke Sewell or Dr. Andrew Fagg (andrewhfagg@gmail.com)  
For questions regarding **Schooner**: Contact OSCER support (support@oscer.ou.edu)  
