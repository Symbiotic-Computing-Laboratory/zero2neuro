#!/bin/bash

# Convert tabular data into TF-Dataset format internally

set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../tests/data_conversion_test

python $ZERO2NEURO_PATH/zero2neuro.py @data1.txt @experiment.txt @network.txt -vvv --force --epochs 2 

