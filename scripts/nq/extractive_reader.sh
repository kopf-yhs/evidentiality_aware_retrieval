#!bin/bash
MODEL_PATH=""
RESULT_PATH=""
OUTPUT_PATH=reader.json

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_extractive_reader.py \
    prediction_results_file=${OUTPUT_PATH} \
    eval_top_docs=[5,20,50] \
    model_file=${MODEL_PATH} \
    dev_files=${RESULT_PATH} \
    train.dev_batch_size=36 \
    passages_per_question_predict=100 \
    encoder.sequence_length=350