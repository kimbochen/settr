#! /usr/bin/env bash

set -o xtrace

# python train_bart.py --lr="1e-6" --n_steps=300
# python train_bart.py --ckpt="bart-lr1.00e-06-step300" --lr="5e-5" --n_steps=600
# python train_bart.py --ckpt="bart-lr5.00e-05-step600" --lr="1e-5" --n_steps=300
# python train_bart.py --ckpt="bart-lr1.00e-05-step300" --lr="3e-6" --n_steps=300

# python train_bart.py --lr="1e-6" --ckpt="facebook/bart-large-cnn" --n_steps=300 --batch_size=8 --grad_acc=2
# python train_bart.py --ckpt="bart-large-cnn-lr1.00e-06-step300" --lr="5e-5" --n_steps=300 --batch_size=8 --grad_acc=2
# python train_bart.py --ckpt="bart-large-cnn-lr5.00e-05-step300" --lr="1e-5" --n_steps=300 --batch_size=8 --grad_acc=2

python train_bart.py --batch_size=16 --grad_acc=2 --lr="5e-5" --ckpt="facebook/bart-base" --n_epochs=10
