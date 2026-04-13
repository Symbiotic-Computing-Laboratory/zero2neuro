---
title: Pickle Test
nav_order: 60
has_children: false
parent: Zero2Neuro Tests
---

# Test: Pickle Data File Examples

This test demonstrates different ways of mixing and matching pickle data configurations for model training.  
Pickle files for Zero2Neuro contain a single object that is a dictionary.  The different data_inputs and data_outputs select the keys of this dictionary

Three key demonstrations:
1. All input columns and all examples are contained within a single dictionary key
2. Examples are split across multiple pickle files
3. Examples are split across multiple pickle files, and different column sets are split across different dictionary keys

Generate the input pickle files for an 4-bit parity problem:
```
python generate_data.py
```


Generated files:
- parity.pkl: all examples (total is 16)
   - ins: full 4-bits
   - outs: output

- parity0.pkl: first 8 examples
- parity1.pkl: next 4 examples
- parity2.pkl: next 3 examples
- parity3.pkl: remaining 1 example

- parityB.pkl: all examples (total is 16)
   - ins0: first 3 columns of the input
   - ins1: reamaining 1 column of the input
   - outs: output
- parityB0.pkl: first 8 examples
   - (same structure as parityB0)
- parityB1.pkl: next 4 examples
- parityB2.pkl: next 3 examples
- parityB3.pkl: remaining 1 example


## Data
- XOR Data: pickle files defined above
- Data configurations: 
   - [data_config.txt](data_config.txt): 
      - Training: parity.pkl (all data)
      - Validation: parity0.pkl (subset of the training set)
   - [data_config2.txt](data_config2.txt): 
      - Training: parity0.pkl (a subset of all data)
      - Validation: parity.pkl (all data)
   - [data_config3.txt](data_config3.txt): 
      - Training: parity0.pkl .. parity3.pkl (all data merged from 4 files)
      - Validation: parity.pkl (all data)
   - [data_config4.txt](data_config4.txt): 
      - Training: parityB.pkl (all data)
      - Validation: none
   - [data_config5.txt](data_config5.txt): 
      - Training: parityB0.pkl .. parityB3.pkl (all data merged from 4 files)
      - Validation: parityB.pkl (all data)
   - [data_config6.txt](data_config6.txt): 
      - Training: parityB0.pkl .. parityB3.pkl (all data merged from 4 files)
      - Validation: parityB.pkl (all data)
      - Testing: parityB.pkl (all data)
   
## Network
- Four binary inputs
- Two hidden layers: 1000 and 100 hidden units
- One binary output
- Network configuration: [network.txt](network.txt)

## Training Details
- [experiment.txt](experiment.txt)

## Training Process: Command Line
Used for testing (assuming that the python environment is already active):
```
python $ZERO2NEURO_PATH/zero2neuro.py @network_config.txt @data_config6.txt @experiment_config.txt -vvv 
```

## Training Process: SLURM Batch File
- File: [batch.sh](batch.sh)

