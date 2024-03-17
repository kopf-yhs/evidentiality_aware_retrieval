#!/bin/bash
EVAL_DATA=multihop_dense_retrieval/scripts/data/hotpot/hotpot_qas_val.json
CORPUS_VECTOR_PATH=multihop_dense_retrieval/index/hotpot_dpr/hotpot_dpr_ours.npy
CORPUS_DICT=multihop_dense_retrieval/scripts/data/hotpot_index/wiki_id2doc.json
MODEL_CHECKPOINT=multihop_dense_retrieval/logs/../model_name/checkpoint_last.pt
PATH_TO_SAVE_RETRIEVAL=multihop_dense_retrieval/scores/hotpot_dpr_ours/

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/common/miniconda3/envs/MDR/lib/

python scripts/eval/eval_mhop_retrieval.py ${EVAL_DATA} ${CORPUS_VECTOR_PATH} ${CORPUS_DICT} ${MODEL_CHECKPOINT} \
    --batch-size 48 \
    --beam-size 100 \
    --topk 10 \
    --shared-encoder \
    --model-name roberta-base \
    --gpu \
    --save-path ${PATH_TO_SAVE_RETRIEVAL}
