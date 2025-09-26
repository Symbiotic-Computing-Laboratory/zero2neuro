# Test: sample weights in tabular data

## Data
- XOR Data: [xor_data.csv](xor_data.csv)
- Data configuration: [data.txt](data.txt)
- The weights are set such that the model should learn to accurately predict the first 3 examples.  The fouth example has no weight, so its error does not factor into the loss (and the model will typically assign a 1 as the output).  However, the metrics take all examples into account, so we expect loss to go to zero, but the metrics will reflect the error in the last example.

## Network
- Two binary inputs
- Two hidden layers: 20 and 10 hidden units
- One binary output
- Network configuration: [network.txt](network.txt)

## Training Details
- [experiment.txt](experiment.txt)

## Training Process: Command Line
Used for testing (assuming that the python environment is already active):
```
./exec.sh
```

## Training Process: SLURM Batch File
- File: [batch.sh](batch.sh)
