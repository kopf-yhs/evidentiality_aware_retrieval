#!/bin/bash
TOP_PATH=""
QA_NAME=trivia_test
CONTEXT_NAME=dpr_wiki

EPOCH=""
CHECKPOINT_PATH=${TOP_PATH}/checkpoint
CHECKPOINT_NAME=""
EMBEDDING_OUTPUT_PATH=${CHECKPOINT_NAME}_e${EPOCH}_on_${CONTEXT_NAME}
SCORE_OUTPUT_PATH=${CHECKPOINT_NAME}_on_${CONTEXT_NAME}

NUM_SHARD=30

for (( i=2*${NUM_SHARD}/6; i<3*${NUM_SHARD}/6; i++ ))
do
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=2 python generate_dense_embeddings.py \
	model_file=${CHECKPOINT_PATH}/${CHECKPOINT_NAME}/dpr_biencoder.${EPOCH} \
	ctx_src=${CONTEXT_NAME} \
	shard_id=${i} num_shards=${NUM_SHARD} \
	batch_size=1792 \
	output_path=${EMBEDDING_OUTPUT_PATH}
done
