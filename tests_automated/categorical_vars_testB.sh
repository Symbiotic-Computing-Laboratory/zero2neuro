#!/bin/bash

# Categorical variable test: translate ints 31/32 into 0/1

set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../tests/categorical_vars

python $ZERO2NEURO_PATH/zero2neuro.py @data2.txt @experiment.txt @network.txt -vvv --force --epochs 2 

