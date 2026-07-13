#!/bin/bash
set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../examples/iris

python $ZERO2NEURO_PATH/zero2neuro.py @network.txt @data.txt @experiment.txt -vvv --force --epochs 2 

