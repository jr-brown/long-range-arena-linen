#!/bin/sh

run(){
    printf "\n\n### Listops | $1 ###\n\n"
    ./dispatch.sh Listops $1 -l
}

run transformer
run local
run longformer
run reformer
run linformer
run sinkhorn
run linear
run bigbird
run synthesizer
run sparse
run performer

