#!/bin/bash

# Import pickle data
# Training: all of the data (merged from several keys from multiple files)
# Validation: all data from one file and multiple keys
# Testing: all data from one file and multiple keys

set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../tests/pickle_test

# Perform the test
python $ZERO2NEURO_PATH/zero2neuro.py @data_config6.txt @experiment_config.txt @network_config.txt -vvv --force --epochs 2 

