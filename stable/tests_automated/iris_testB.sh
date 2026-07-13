#!/bin/bash
# Check sklearn implementations: SGD Classifier
set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../examples/iris

python $ZERO2NEURO_PATH/zero2neuro.py @network_skl1.txt @data.txt @experiment_skl.txt -vvv --force --epochs 2 

