#!/bin/bash

# Testing different tabular configurations: key columns locations
# Specific list of columns

set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../tests/tabular_test

# Perform the test
python $ZERO2NEURO_PATH/zero2neuro.py @data2.txt @experiment.txt @network.txt -vvv --force --epochs 2 

