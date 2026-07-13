#!/bin/bash

# Categorical variable test with pickle files: translate strings F/T into 0/1

set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../tests/categorical_vars

python $ZERO2NEURO_PATH/zero2neuro.py @data_p.txt @experiment.txt @network.txt -vvv --force --epochs 2 

