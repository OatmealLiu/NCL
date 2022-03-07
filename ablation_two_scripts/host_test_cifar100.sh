#!/usr/bin/env bash

python -W ignore two_NCL_incd_train_cifar100.py \
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
        --num_unlabeled_classes1 10 \
        --num_unlabeled_classes2 10 \
        --num_labeled_classes 80 \
        --dataset_name cifar100 \
        --seed 5 \
        --model_name NCL_1st_cifar100 \
        --mode train \
        --bce_type cos \
        --wandb_mode offline \
        --wandb_entity oatmealliu \
        --step first

  python -W ignore two_NCL_incd_train_cifar100.py \
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
        --num_unlabeled_classes1 10 \
        --num_unlabeled_classes2 10 \
        --num_labeled_classes 80 \
        --dataset_name cifar100 \
        --seed 5 \
        --model_name NCL_2nd_cifar100 \
        --mode train \
        --bce_type cos \
        --wandb_mode offline \
        --wandb_entity oatmealliu \
        --step second \
        --first_step_dir ./data/experiments/two_NCL_incd_train_cifar100/first_NCL_1st_cifar100.pth


