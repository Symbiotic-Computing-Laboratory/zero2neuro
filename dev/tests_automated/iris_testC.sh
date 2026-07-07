#!/bin/bash
# Check sklearn implementations: Decision Tree classifier
set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../examples/iris

python $ZERO2NEURO_PATH/zero2neuro.py @network_skl2.txt @data.txt @experiment_skl.txt -vvv --force --epochs 2 

