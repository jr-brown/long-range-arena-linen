#!/bin/sh

export PYTHONPATH="$(pwd)":"$PYTHONPATH"

# Required args
TASK=$1
MODEL=$2
shift 2

# Set defaults
LOG=0
CFG_PATH="configs"

# Process options
while getopts 'lt' opt
do
    case $opt in
        l) LOG=1 ;;
        t) CFG_PATH="configs/test_configs" ;;
    esac
done

if [ $LOG = 1 ]; then
    XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1 python -u lra_benchmarks/$TASK/train.py \
          --config=lra_benchmarks/$TASK/$CFG_PATH/${MODEL}_base.py \
          --model_dir=trained_models/$TASK/$MODEL \
          > logs/${TASK}_$MODEL.txt 2>&1
else
    XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1 python -u lra_benchmarks/$TASK/train.py \
          --config=lra_benchmarks/$TASK/$CFG_PATH/${MODEL}_base.py \
          --model_dir=trained_models/$TASK/$MODEL
fi

