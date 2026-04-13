---
title: Categorical Variables Test
nav_order: 10
has_children: false
parent: Zero2Neuro Tests
---

# Zero2Neuro Test: Categorical Variables

Testing the translation of categorical variables into integers

## Case 1
- Translate T/F into 1 and 0.
- [data.txt](data.txt)
- [xor_data.csv](xor_data.csv)

## Case 2
- Output class values are 31 or 32.  Translate 31/32 into 1 and 0 (this is a test of having integers in the column)
- [data2.txt](data2.txt)
- [xor_data2.csv](xor_data2.csv)

## Case 3
- Using pickle files as inputs
- Translate T/F into 1 and 0.
- [data_p.txt](data_p.txt)
- data_p.pkl: Converted from data.csv using csv2python.py (see bin dir)

## Case 4
- Using pickle files as inputs
- Output class values are 31 or 32.  Translate 31/32 into 1 and 0 (this is a test of having integers in the column)
- [data_p2.txt](data_p2.txt)
- data_p2.pkl: Converted from data2.csv using csv2python.py (see bin dir)

## Network
- Two binary inputs
- Two hidden layers: 20 and 10 hidden units
- One binary output
- Network configuration: [network.txt](network.txt)

## Training Details
- [experiment.txt](experiment.txt)

## Training Process: SLURM Batch File
- File: [batch.sh](batch.sh)

## Training Command (example)
```
python $ZERO2NEURO_PATH/zero2neuro.py @network.txt @data.txt @experiment.txt -vvv 
```
