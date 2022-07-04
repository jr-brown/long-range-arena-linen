#!/bin/sh

run(){
    printf "\n\n### Text Classification | $1 ###\n\n"
    ./dispatch.sh text_classification $1
}

run transformer
run local
run reformer
run linformer
run longformer

