

# <div style="display: flex; justify-content: center; align-items: center; height: 300px"> <img SRC="images/zero2neuro2.png" height="250" style="margin-right: 100px;" alt="Zero2Neuro Icon"> Zero2Neuro </div>



## Authors

Andrew H. Fagg (andrewhfagg@gmail.com)  
Mel Wilson Reyes  
Luke Sewell  

## Requirements

- Python environment with Keras3 and Tensorflow 2.xx


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

- [Zero2Neuro Documentation](docs/index.md)
- [Zero2Neuro Presentation](https://docs.google.com/presentation/d/12ZBsMVq-6mW498PQZfDNP1_Mfu_3sIMWwnczJN84O5I/edit?usp=sharing)
