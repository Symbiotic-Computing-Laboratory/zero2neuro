#!/bin/bash
set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../examples/sentiment140

python $ZERO2NEURO_PATH/zero2neuro.py @demo_network_gru.txt @demo_data.txt @demo_experiment.txt -vvv --force --epochs 2 

