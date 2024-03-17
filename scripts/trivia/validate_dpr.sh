#!/bin/bash

EPOCH=""
TOP_PATH=""
CHECKPOINT_NAME=""
CHECKPOINT_PATH=${TOP_PATH}/checkpoint/${CHECKPOINT_NAME}/dpr_biencoder.${EPOCH}

QA_NAME=trivia_test
CONTEXT_NAME=dpr_wiki

OUTPUT_PATH=${CHECKPOINT_NAME}_e${EPOCH}_on_${CONTEXT_NAME}
EMBEDDING_PATH=${TOP_PATH}/embeddings/${CHECKPOINT_NAME}_e${EPOCH}_on_${CONTEXT_NAME}

CUDA_VISIBLE_DEVICES=0 python dense_retriever.py \
	model_file=${CHECKPOINT_PATH} \
	qa_dataset=${QA_NAME} \
	ctx_datasets=[${CONTEXT_NAME}] \
	encoded_ctx_files=[\"${EMBEDDING_PATH}/test_embeddings.pkl_*\"] \
	output_path=${OUTPUT_PATH} \
	out_file=scores.json