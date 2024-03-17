#!/bin/bash
EVAL_DATA=multihop_dense_retrieval/scripts/data/hotpot/hotpot_qas_val.json
CORPUS_VECTOR_PATH=multihop_dense_retrieval/index/hotpot_aadpr/hotpot_aadpr_ours_epoch18.npy
CORPUS_DICT=multihop_dense_retrieval/scripts/data/hotpot_index/wiki_id2doc.json
MODEL_CHECKPOINT=multihop_dense_retrieval/savepath/aadpr_checkpoint_last_epoch18.pt
PATH_TO_SAVE_RETRIEVAL=multihop_dense_retrieval/scores/hotpot_aadpr_last/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/common/miniconda3/envs/MDR/lib/

python scripts/eval/eval_mhop_retrieval.py ${EVAL_DATA} ${CORPUS_VECTOR_PATH} ${CORPUS_DICT} ${MODEL_CHECKPOINT} \
    --batch-size 40 \
    --beam-size 20 \
    --topk 10 \
    --shared-encoder \
    --model-name roberta-base \
    --gpu \
    --save-path ${PATH_TO_SAVE_RETRIEVAL}
