#!/bin/bash

for TRAIN in 50 100 150 200 250 300 350 400
do
    for FOLD in 1 2 3 4 5
    do
        python code/experiment01.py --input-data data/B-59.850.lst --gpu 0 --train-limit $TRAIN --log exp01_train_${TRAIN}_${FOLD}-5
    done
done
