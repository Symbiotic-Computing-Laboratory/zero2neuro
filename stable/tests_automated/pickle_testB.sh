#!/bin/bash

# Import pickle data
# Training: subset of the data; validation: all of the data

set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../tests/pickle_test

# Perform the test
python $ZERO2NEURO_PATH/zero2neuro.py @data_config2.txt @experiment_config.txt @network_config.txt -vvv --force --epochs 2 

