#!/bin/bash
CORPUS_PATH=multihop_dense_retrieval/scripts/data/hotpot_index/wiki_id2doc.json
MODEL_CHECKPOINT=multihop_dense_retrieval/savepath/aadpr_checkpoint_last_epoch18.pt
SAVE_PATH=multihop_dense_retrieval/index/hotpot_aadpr/hotpot_aadpr_ours_epoch18.npy

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/encode_corpus.py \
    --do_predict \
    --predict_batch_size 8192 \
    --model_name roberta-base \
    --predict_file ${CORPUS_PATH} \
    --init_checkpoint ${MODEL_CHECKPOINT} \
    --embed_save_path ${SAVE_PATH} \
    --preprocessed \
    --fp16 \
    --max_c_len 300 \
    --num_workers 20 