#!/bin/bash

# Testing the use of non-default rows and columns in xlsx file
# Test column list
# Test training and validation data coming from different sheets


set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../tests/tabular_xlsx_test

# Perform the test
python $ZERO2NEURO_PATH/zero2neuro.py @data2.txt @experiment.txt @network.txt -vvv --force --epochs 2 

