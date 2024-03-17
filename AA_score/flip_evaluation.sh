#!/bin/bash

python flip_evaluation.py \
    --score_file ../outputs/validation/checkpoint_name/scores.json \
    --answer qrels_gold_passages.csv \
    --masked_answer qrels_masked_gold_passages.csv
    