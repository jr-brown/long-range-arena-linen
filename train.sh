#!/bin/sh

export PYTHONPATH="$(pwd)":"$PYTHONPATH"

python -u lra_benchmarks/text_classification/train.py \
      --config=lra_benchmarks/text_classification/configs/transformer_base.py \
      --model_dir=models/text_classification \
      --task_name="yelp_reviews" \
      >> train_log.txt 2>&1

