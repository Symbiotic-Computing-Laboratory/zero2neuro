#!/bin/bash
set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../examples/breast_cancer

python $ZERO2NEURO_PATH/zero2neuro.py @network_config.txt @data_config.txt @experiment_config.txt -vvv --force --epochs 2 

