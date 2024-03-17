#!/bin/bash

# Note that evidence annotation process can be done in parallel, by assigning each shard to different devices.

DATA_PATH=
NUM_SHARDS=4

for (( i=0; i<${NUM_SHARDS}; i++))
do
    CUDA_VISIBLE_DEVICES=0 python run_flan_t5.py \
        --data_path ${DATA_PATH} \
        --model_file allenai/unifiedqa-v2-t5-3b-1251000 \
        --shard_idx 0 \
        --num_shard ${NUM_SHARDS} \
done