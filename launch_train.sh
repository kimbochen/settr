#! /usr/bin/env bash

set -o xtrace

# python train_bart.py --lr="1e-6" --n_steps=300
# python train_bart.py --ckpt="bart-lr1.00e-06-step300" --lr="5e-5" --n_steps=600
# python train_bart.py --ckpt="bart-lr5.00e-05-step600" --lr="1e-5" --n_steps=300
# python train_bart.py --ckpt="bart-lr1.00e-05-step300" --lr="3e-6" --n_steps=300

# python train_bart.py --lr="1e-6" --ckpt="facebook/bart-large-cnn" --n_steps=300 --batch_size=8 --grad_acc=2
# python train_bart.py --ckpt="bart-large-cnn-lr1.00e-06-step300" --lr="5e-5" --n_steps=300 --batch_size=8 --grad_acc=2
# python train_bart.py --ckpt="bart-large-cnn-lr5.00e-05-step300" --lr="1e-5" --n_steps=300 --batch_size=8 --grad_acc=2

# python train_bart.py --batch_size=16 --grad_acc=2 --lr="5e-5" --ckpt="facebook/bart-base" --n_epochs=10
# python train_bart.py --batch_size=16 --grad_acc=2 --lr="5e-5" --ckpt="runs/bart-base-lr5.00e-05-10epochs" --n_epochs=10 --save_dir="runs/train-longer"
# python train_bart.py --batch_size=16 --grad_acc=2 --lr="5e-5" --ckpt="runs/train-longer" --n_epochs=30 --save_dir="runs/overfit"

# python train_bart.py --ckpt="facebook/bart-base" --batch_size=16 --lr="5e-5" --n_steps=4000 --eval_freq=200 --save_dir="runs/bart-base-first-run"
# python train_bart.py --ckpt="facebook/bart-base" --batch_size=16 --lr="5e-5" --warmup=800 --n_steps=4000 --eval_freq=200 --save_dir="runs/bart-base-warmup"
