#!/bin/bash

# Import pickle data
# Training: all of the data (merged from several files); validation: all of the data

set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../tests/pickle_test

# Perform the test
python $ZERO2NEURO_PATH/zero2neuro.py @data_config3.txt @experiment_config.txt @network_config.txt -vvv --force --epochs 2 

