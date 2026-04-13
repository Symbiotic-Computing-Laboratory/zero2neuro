---
title: Data Conversion Test
nav_order: 40
has_children: false
parent: Zero2Neuro Tests
---

# Test: Internal Data Format: TF-Dataset

After each table is loaded, it is converted to a TF-Dataset for subsequent processing

## Data
- XOR Data: [xor_data.xlsx](xor_data.xlsx): 2 sheets
- Data configuration: [data1.txt](data1.txt): Use a single sheet for one table
- Data configuration: [data2.txt](data2.txt): Use two sheets: one for training, the other for validation

## Network
- Two binary inputs
- Two hidden layers: 20 and 10 hidden units
- One binary output
- Network configuration: [network.txt](network.txt)

## Training Details
- [experiment.txt](experiment.txt)

## Training Process: Command Line
Execute from the tests/tabular_xlsx_test/ directory:
```
./exec1.sh

OR

./exec2.sh

```

## Training Process: SLURM Batch File
- File: [batch.sh](batch.sh)
