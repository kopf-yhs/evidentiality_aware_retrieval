#!/bin/bash

EPOCH=""

CHECKPOINT_PATH=""
CHECKPOINT_NAME=""
CONTEXT_NAME=nq_gold_info_cf
EMBEDDING_OUTPUT_PATH=${CHECKPOINT_NAME}_on_dpr_wiki

CUDA_VISIBLE_DEVICES=0 python generate_dense_embeddings.py \
	model_file=${CHECKPOINT_PATH}/${CHECKPOINT_NAME}/dpr_biencoder.${EPOCH} \
	ctx_src=${CONTEXT_NAME} \
	shard_id=0 num_shards=1 \
	batch_size=1792 \
	out_file=test_embeddings.pkl_gold \
	output_path=${EMBEDDING_OUTPUT_PATH}
