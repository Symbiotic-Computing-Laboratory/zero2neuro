#!/bin/bash

# Import pickle data
# Training: all data; validation: subset of the data

set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../tests/pickle_test

# Generate the pickle files (all other tests in this series rely on this)
python generate_data.py

# Perform the test
python $ZERO2NEURO_PATH/zero2neuro.py @data_config.txt @experiment_config.txt @network_config.txt -vvv --force --epochs 2 

