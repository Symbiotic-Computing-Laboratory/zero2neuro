---
title: Full Example
nav_order: 1
parent: Zero2Neuro
---
# Full Zero2Neuro Example: Exclusive OR

For this example, we will create a fully-connected network to learn a
simple binary function -- the Exclusive OR (XOR) function.  This
function has 4 possible cases (one on each row):

| In 0 | In 1 | Out 0 |
|------|------|-------------|
|   0  |   0  |      0      |
|   1  |   0  |      1      |
|   0  |   1  |      1      |
|   1  |   1  |      0      |

_In 0_ and _In 1_ are the two inputs into the network, and the _Out 0_
column shows the correct value that the network should produce for
each of the cases.

## Data Configuration
We specify the data with two different files:
- The XOR data itself is specified using a CSV file format: [xor_data.csv](../../examples/xor/xor_data.csv)
- The data configuration file states how the CSV file will be
interpreted: [data.txt](../../examples/xor/data.txt)

data.txt looks like this:
```
# Data is from a table
--data_format=tabular
--data_file=xor_data.csv

# Our one file becomes our training set
--data_set_type=fixed

# Two inputs
--data_inputs
In 0
In 1

# One output
--data_outputs
Out 0

```

Notes:
- In any of the configuration files, characters following the # symbol
are ignored.  Typically, we use this feature to add textual
explanations about the following lines.

- Given that there is a single input file and --data_set_type=fixed,
all of the examples in the file will be used for training.  We do not
have a validation or test data set in this example.

___
## Network Configuration

In this example we are implementing a _fully connected_ network with
two _hidden_ (intermediate) layers of neurons before the network
produces its output.  For XOR, at least one hidden layer is required
to solve the problem.

The [network.txt configuration file](../../examples/xor/network.txt)
looks like this:
```
# Fully connected network
--network_type=fully_connected

# Two inputs
--input_shape
2

# Two hidden layers
--number_hidden_units
20
10
--hidden_activation=elu

# One output unit
--output_shape
1
--output_activation=sigmoid

```

Here, we declare the full structure of the network:
- __--input_shape__: each example has two inputs; these form a vector (or
1-dimensional tensor)
- __--number_hidden_units__: a list of integers (20 and 10 in this
case) that declare how many hidden units there are in each of the two
hidden layers.  All input units connect to all units in the first
hidden layer; all units in the first hidden layer connect to all units
in the second hidden layer
- __--hidden_activation__: declares which non-linear function is
used for each of the hidden units
- __--output_shape__: the output in this example is a single value
(this must match the number of __--data_outputs__)
- __--output_activation__: declares the non-linear function for the
output unit.  Here, we choose sigmoid because we want our output
values to fall within the 0...1 range.

___
## Experiment Configuration

The final configuration file describes the details of the training and
reporting process.


The [experiment.txt configuration file](../../examples/xor/experiment.txt)
looks like this:
```
# Name used for files and wandb
--experiment_name=xor

# Loss function is what we minimize during training; mse=Mean Squared Error
--loss=mse

# We might also care about other metrics; mae=Mean Absolute Error
--metrics
mae
mse

# How fast we adjust the parameters
--learning_rate=0.001

# Maximum number of steps for training
--epochs=5000

# Stop the training early if no improvement is observed
# Note: we typically monitor val_loss when we have a validation data set
--early_stopping
--early_stopping_monitor=loss
--early_stopping_patience=2000

# Where to place trained networks, reports, and stored results
--results_path=./results

# Expanded file name
--output_file_base={args.experiment_name}_R{args.data_rotation:02d}

# Save the trained model to a file
--save_model

# Create a picture of the model
--render_model

# Save the training set results
--log_training_set

# Report the results to a XLSX file
--report
--report_training
--report_training_ins
```

Key arguments:
- __--loss__: this defines the function that will be minimized during
training
- __--metrics__: a list of functions that measure model performance
- __--learning_rate__: defines how quickly the training process will
proceed (but too high can be devastating)
- __--epochs__: the maximum number of training steps (subject to early
stopping)
- __--save_model__: write the final Keras3 model to a file.  This can
be loaded at a later time
- __--render_model__: create a picture of your model and write it to a
file
- __log_training_set__: write the details of training set inputs and
corresponding network outputs to a pickle file for later analysis
- __report__: generate a report in XLSX file format of the performance
of the trained model
- __report_training__: include in the report a sheet that contains
the true and model output for each example
- __report_training_ins__: include on the above sheet the example
inputs

___
## Execution in the Bash Shell

1. Activate Python Environment.  The details will vary depending on
your system.  On the OU Supercomputer, this is done as follows:

```
. /home/fagg/tf_setup.sh
conda activate dnn
```

2. Configure your NEURO_REPOSITORY_PATH.  The location will depend
your local configuration.  On the OU Supercomputer:

```
export NEURO_REPOSITORY_PATH=/home/fagg/zero2neuro
```

3. Change your current working directory to the location of the
example.  If you are using the XOR example:

```
cd examples/xor
```

4. Execute Your Experiment

```
python $NEURO_REPOSITORY_PATH/zero2neuro/src/zero2neuro.py @network.txt @data.txt @experiment.txt -vvv
```

- @network.txt: load the arguments from the network.txt file
- @data.txt: load the arguments from data.txt
- @experiment.txt: load experiment.txt
- -vvv: be very verbose about reporting the state of the experiment

For this experiment, the Zero2Neuro engine will generate [text
output](full_example_output.txt)

___
## Training Outputs
The outputs are placed in the __results/__ subdirectory:
- xor_R00_model.keras: reloadable model
- xor_R00_model_plot.png: image that describes the model
- xor_R00_report.xlsx: report file in excel format
- xor_R00_results.pkl: pickle file that contains the model training
information and predictions for the training set examples.  This file
can be read using a Python program

___

## References
- [Full XOR Description](../../examples/xor/README.md)

