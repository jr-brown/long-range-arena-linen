#!/bin/sh

run(){
    printf "\n\n### Listops | $1 ###\n\n"
    ./dispatch.sh listops $1 -l
}

run linear
run sparse

