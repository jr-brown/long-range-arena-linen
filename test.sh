#!/bin/sh

export PYTHONPATH="$(pwd)":"$PYTHONPATH"

python lra_benchmarks/text_classification/train.py \
      --config=lra_benchmarks/text_classification/configs/transformer_base_test.py \
      --model_dir=models/text_classification \
      --task_name="yelp_reviews" \

