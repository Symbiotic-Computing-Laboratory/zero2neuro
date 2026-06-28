#!/bin/bash
set -euo pipefail

export ZERO2NEURO_PATH=$NEURO_REPOSITORY_PATH/zero2neuro/src

cd ../examples/mesonet

python $ZERO2NEURO_PATH/zero2neuro.py @data_config.txt @experiment_config.txt @network_config.txt @report.txt @classification_override.txt -vvv --force --epochs 2 

