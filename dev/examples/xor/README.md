# Example: Implementing the Exclusive OR (XOR) Function

## Data
- XOR Data: [xor_data.csv](xor_data.csv)
- Data configuration: [data.txt](data.txt)

## Network
- Two binary inputs
- Two hidden layers: 20 and 10 hidden units
- One binary output
- Network configuration: [network.txt](network.txt)

## Training Details
- [experiment.txt](experiment.txt)

## Training Process: Command Line
Execute from the examples/xor/ directory:
```
python $NEURO_REPOSITORY_PATH/zero2neuro/src/zero2neuro.py @network.txt @data.txt @experiment.txt -vvv

```

## Training Process: SLURM Batch File
- File: [batch.sh](batch.sh)
