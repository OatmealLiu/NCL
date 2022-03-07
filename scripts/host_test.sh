#!/usr/bin/env bash

python -W ignore ncl_all.py \
        --dataset_root ./data/datasets/CIFAR/ \
        --exp_root ./data/experiments/ \
        --warmup_model_dir ./data/experiments/supervised_learning_wo_ssl/warmup_cifar100_resnet_wo_ssl.pth \
        --lr 0.1 \
        --gamma 0.1 \
        --weight_decay 1e-4 \
        --step_size 170 \
        --batch_size 128 \
        --epochs 1 \
        --rampup_length 150 \
        --rampup_coefficient 50 \
        --num_labeled_classes 80 \
        --num_unlabeled_classes 20 \
        --dataset_name cifar100 \
        --seed 5 \
        --model_name resnet_cifar100_ncl \
        --mode train \
        --bce_type cos \
        --wandb_mode offline \
        --wandb_entity oatmealliu



