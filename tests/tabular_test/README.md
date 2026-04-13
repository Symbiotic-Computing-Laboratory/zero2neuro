---
title: Tabular Test
nav_order: 20
has_children: false
parent: Zero2Neuro Tests
---

# Test: different header row and data columns using XOR problem

## Data
- XOR Data: [xor_data.csv](xor_data.csv)
- Data configuration: [data1.txt](data1.txt): provides a range of columns to extract from the CSV file
- Data configuration: [data2.txt](data2.txt): provides a list of columns to extract

## Network
- Two binary inputs
- Two hidden layers: 20 and 10 hidden units
- One binary output
- Network configuration: [network.txt](network.txt)

## Training Details
- [experiment.txt](experiment.txt)

## Training Process: Command Line
Execute from the tests/tabular_test/ directory:
```
python $NEURO_REPOSITORY_PATH/zero2neuro/src/zero2neuro.py @network.txt @data.txt @experiment.txt -vvv
```

## Training Process: SLURM Batch File
- File: [batch.sh](batch.sh)
