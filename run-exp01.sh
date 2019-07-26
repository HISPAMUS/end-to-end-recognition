#!/bin/bash

for FOLD in 1 2 3 4 5
do
    for TRAIN in 50 100 150 200 250 300 350 400
    do
        python code/experiment01.py --input-data data/B-59.850.lst --gpu 0 --train-limit $TRAIN --log exp01_train_${TRAIN}_${FOLD}-5
    done
    python code/experiment01.py --input-data data/B-59.850.lst --gpu 0 --log exp01_train_FULL_${FOLD}-5
done
