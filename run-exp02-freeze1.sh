#!/bin/bash

SEEDS=(123 234 345 456 567)
FREEZE=2

for FOLD in 1 2 3 4 5
do
    SEED=${SEEDS[$((FOLD-1))]}
    for TRAIN in 50 100 150 200 250 300 350 400
    do
       python code/experiment02.py --input-data data/B-59.850.lst --seed $SEED --gpu 1 --train-limit $TRAIN --freeze $FREEZE --log exp02_freeze${FREEZE}_train_${TRAIN}_${FOLD}-5
    done
    python code/experiment02.py --input-data data/B-59.850.lst --seed $SEED --gpu 1 --freeze $FREEZE --log exp02_freeze${FREEZE}_train_FULL_${FOLD}-5
done
