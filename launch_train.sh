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

# python train_bart.py --ckpt="facebook/bart-base" --batch_size=16 --grad_acc=2 --lr="2e-5" --warmup=200 --n_steps=800 --eval_freq=40 --save_dir="runs/bart-base-first-run"
# python train_bart.py --ckpt="runs/bart-base-first-run" --batch_size=16 --grad_acc=2 --lr="2e-5" --n_steps=800 --eval_freq=40 --save_dir="runs/bart-base-overfit"
# python train_bart.py --ckpt="facebook/bart-base" --batch_size=16 --grad_acc=2 --lr="2e-5" --warmup=400 --n_steps=800 --eval_freq=40 --save_dir="runs/bart-base-more-warmup"

# python train_bart.py --ckpt="facebook/bart-base" --batch_size=16 --grad_acc=2 --lr="2e-5" --warmup=400 --n_steps=1200 --eval_freq=40 --save_dir="runs/bart-base-balanced-data-train-longer"
# python train_bart.py --ckpt="t5-base" --batch_size=8 --grad_acc=4 --lr="2e-5" --warmup=400 --n_steps=1200 --eval_freq=40 --save_dir="runs/t5-base-balanced-data-train-longer"
# python train_bart.py --ckpt="bigscience/T0_3B" --batch_size=1 --grad_acc=32 --lr="2e-5" --warmup=400 --n_steps=1200 --eval_freq=60 --save_dir="runs/t0-3b-train-longer"


randomSeeds=(39 85 99 42 27)

# How does balancing dataset affect performance
# for i in {0..4}
# do
#     python train_model.py \
#         --ckpt "facebook/bart-base" --exp_name "allsumm_${i}" \
#         --seed ${randomSeeds[i]} --dd "allsumm_concat" \
#         --train_steps 1200 --lr "2e-5" --warmup 400 \
#         --batch_size 16 --grad_acc 2 \
#         --eval_freq 60 
# 
#     python train_model.py \
#         --ckpt "facebook/bart-base" --exp_name "balanced_${i}" \
#         --seed ${randomSeeds[i]} --dd "balanced" \
#         --train_steps 1200 --lr "2e-5" --warmup 400 \
#         --batch_size 16 --grad_acc 2 \
#         --eval_freq 60 
# done

# Trying out different models
for i in {0..4}
do
    python train_model.py \
        --ckpt "t5-base" --exp_name "allsumm_${i}_t5" \
        --seed ${randomSeeds[i]} --dd "allsumm_concat" \
        --train_steps 1200 --lr "2e-5" --warmup 400 \
        --batch_size 8 --grad_acc 4 \
        --eval_freq 60 

    python train_model.py \
        --ckpt "t5-base" --exp_name "balanced_${i}_t5" \
        --seed ${randomSeeds[i]} --dd "balanced" \
        --train_steps 1200 --lr "2e-5" --warmup 400 \
        --batch_size 8 --grad_acc 4 \
        --eval_freq 60 
done
