#!/bin/bash

TRAIN_DATA_PATH=multihop_dense_retrieval/hotpot_train_counterfactual.json
DEV_DATA_PATH=multihop_dense_retrieval/scripts/data/hotpot/hotpot_dev_with_neg_v0.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 scripts/train_mhop.py \
    --prefix hotpot_dpr \
    --predict_batch_size 3000 \
    --model_name roberta-base \
    --train_batch_size 15 \
    --learning_rate 2e-5 \
    --fp16 \
    --train_file ${TRAIN_DATA_PATH} \
    --predict_file ${DEV_DATA_PATH}  \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 300 \
    --max_q_len 70 \
    --max_q_sp_len 350 \
    --shared-encoder \
    --warmup-ratio 0.1 \
    --do_train \