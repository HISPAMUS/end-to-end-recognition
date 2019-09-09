#!/bin/bash

SEEDS=(123 234 345 456 567)

for FOLD in 1 2 3 4 5
do
    SEED=${SEEDS[$((FOLD-1))]}
    for TRAIN in 50 100 150 200 250 300 350 400
    do
       python code/experiment01.py --input-data data/B-59.850.lst --seed $SEED --gpu 0 --train-limit $TRAIN --log exp01_train_${TRAIN}_${FOLD}-5
    done
    python code/experiment01.py --input-data data/B-59.850.lst --seed $SEED --gpu 0 --log exp01_train_FULL_${FOLD}-5
done
