#!/bin/bash

TOP_PATH=""
CHECKPOINT_NAME=""
INPUT_FILE=""
OUTPUT_PATH=""

CHECKPOINT_PATH=${TOP_PATH}/checkpoint/${CHECKPOINT_NAME}

CUDA_VISIBLE_DEVICES=0 python compare_counterfactual_samples.py \
	model_file=${CHECKPOINT_PATH} \
	input_file=${INPUT_FILE} \
	output_path=${OUTPUT_PATH} \
	output_remove_percentage=True \
	use_title=True \
