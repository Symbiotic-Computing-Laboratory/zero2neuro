#!/bin/bash

# Test random-stratify assignment of samples to folds

set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../examples/amino

python $ZERO2NEURO_PATH/zero2neuro.py @data.txt @experiment.txt @network_rnn_complex.txt @classification_override2.txt  -vvv --force --epochs 2 

