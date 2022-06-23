#!/bin/sh

PYTHONPATH="$(pwd)":"$PYTHONPATH" python lra_benchmarks/text_classification/train.py \
      --config=lra_benchmarks/text_classification/configs/transformer_base.py \
      --model_dir=/tmp/text_classification \
      --task_name="yelp_reviews" \

