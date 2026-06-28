#!/bin/bash

# Testing the use of sample weights

set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../tests/tabular_weights_test

# Perform the test
python $ZERO2NEURO_PATH/zero2neuro.py @data.txt @experiment.txt @network.txt -vvv --force --epochs 2 

