#!/usr/bin/env bash

python -W ignore ncl_cifar.py \
        --dataset_root ./data/datasets/CIFAR/ \
        --exp_root ./data/experiments/ \
        --warmup_model_dir ./data/experiments/pretrained/supervised_learning/resnet_rotnet_cifar10.pth \
        --lr 0.1 \
        --gamma 0.1 \
        --weight_decay 1e-4 \
        --step_size 170 \
        --batch_size 128 \
        --epochs 200 \
        --rampup_length 50 \
        --rampup_coefficient 5.0 \
        --num_labeled_classes 5 \
        --num_unlabeled_classes 5 \
        --dataset_name cifar10 \
        --seed 5 \
        --model_name resnet_cifar10_ncl \
        --mode train \
        --bce_type cos
