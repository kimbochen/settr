#! /usr/bin/env bash

# python train_bart.py --lr="1e-6" --n_steps=300
python train_bart.py --ckpt_name='bart-lr1.00e-06-step300' --lr="5e-5" --n_steps=600
python train_bart.py --ckpt_name='bart-lr5.00e-05-step600' --lr="1e-5" --n_steps=300
