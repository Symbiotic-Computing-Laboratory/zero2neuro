# Getting Started

Assumptions
- TODO
- You have git installed
- You have a python enviroment set up with Keras3 and TensorFlow
- You have a basic conceptual understanding of DNNs (If this isn't an assumption we should include a markdown that explains the basic of deep neural networks)

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

**Supercomputer**
  
Zero2Neuro was built with supercomputer usage in mind. For information about OU's supercomputer Schooner and how to utilize it visit https://www.ou.edu/oscer. Examples provided in [/examples](../examples/) contain `batch.sh` files to use as examples for running jobs on Schooner.  