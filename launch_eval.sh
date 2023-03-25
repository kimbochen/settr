#! /usr/bin/env bash

set -o xtrace

python eval_bart.py --ckpt=$1 --split=val --batch_size=16
python eval_bart.py --ckpt=$1 --split=test --batch_size=16
