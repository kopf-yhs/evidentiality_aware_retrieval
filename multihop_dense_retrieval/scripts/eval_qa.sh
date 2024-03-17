#!/bin/bash

python scripts/train_qa.py \
    --do_predict \
    --predict_batch_size 800 \
    --model_name google/electra-large-discriminator \
    --fp16 \
    --predict_file multihop_dense_retrieval/gold_mdr_reader_input.json \
    --max_seq_len 512 \
    --max_q_len 64 \
    --init_checkpoint scripts/models/qa_electra.pt \
    --sp-pred \
    --max_ans_len 30 \
    --save-prediction hotpot_val_gold_dpr_mdr_top20.json