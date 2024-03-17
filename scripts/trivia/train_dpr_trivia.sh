#!/bin/bash

CHECKPOINT=""

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HYDRA_FULL_ERROR=1 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 \
    train_dense_encoder.py \
    train=biencoder_custom \
    train.batch_size=16 \
    train_datasets=[trivia_train] \
    dev_datasets=trivia_dev \
    train.num_train_epochs=40 \
    train.other_negatives=0 \
    train.hard_negatives=1 \
    output_dir=$CHECKPOINT