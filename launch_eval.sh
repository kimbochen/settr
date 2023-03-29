#! /usr/bin/env bash

set -o xtrace

randomSeeds=(39 85 99 42 27)

for i in {0..4}
do
    python eval_model.py --seed ${randomSeeds[i]} --ckpt "$1_${i}" --split "val" --batch_size 16
done

for i in {0..4}
do
    python eval_model.py --seed ${randomSeeds[i]} --ckpt "$1_${i}" --split "test" --batch_size 16
done
