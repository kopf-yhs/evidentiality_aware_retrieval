#!/bin/bash
RUN_ID=hotpot_dpr
TRAIN_DATA_PATH=multihop_dense_retrieval/hotpot_train_counterfactual.json
DEV_DATA_PATH=multihop_dense_retrieval/scripts/data/hotpot/hotpot_dev_with_neg_v0.json
CHECKPOINT_PT=multihop_dense_retrieval/logs/../model_name/checkpoint_last.pt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/train_momentum.py -m torch.distributed.launch --nproc_per_node=8 \
    --do_train \
    --prefix ${RUN_ID} \
    --predict_batch_size 3000 \
    --model_name roberta-base \
    --train_batch_size 120 \
    --learning_rate 1e-5 \
    --fp16 \
    --train_file ${TRAIN_DATA_PATH} \
    --predict_file ${DEV_DATA_PATH}  \
    --seed 16 \
    --eval-period -1 \
    --max_c_len 300 \
    --max_q_len 70 \
    --max_q_sp_len 350 \
    --momentum \
    --k 76800 \
    --m 0.999 \
    --temperature 1 \
    --init-retriever ${CHECKPOINT_PT}