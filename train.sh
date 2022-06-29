#!/bin/sh

export PYTHONPATH="$(pwd)":"$PYTHONPATH"

XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1 python -u lra_benchmarks/text_classification/train.py \
      --config=lra_benchmarks/text_classification/configs/transformer_base.py \
      --model_dir=models/text_classification \
      --task_name="yelp_reviews" \

