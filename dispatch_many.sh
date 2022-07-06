#!/bin/sh

run(){
    printf "\n\n### Text Classification | $1 ###\n\n"
    ./dispatch.sh text_classification $1 -l
}

run linear
run bigbird
run synthesizer
run sparse
run performer
run sinkhorn

