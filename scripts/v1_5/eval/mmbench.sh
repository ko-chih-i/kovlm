#!/bin/bash

SPLIT="mmbench_dev_20230712"

#python -m llava.eval.model_vqa_mmbench \
#    --model-path /mnt/tqnas/ali/home/lubin.fan/ckpts/llava-v1.5-7b \
#    --question-file ./playground/data/mmbench/$SPLIT.tsv \
#    --answers-file ./playground/data/mmbench/answers/$SPLIT/llava-7b.jsonl \
#    --single-pred-prompt \
#    --temperature 0 \
#    --conv-mode vicuna_v1

#mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/mmbench/answers_upload/$SPLIT \
    --experiment llava-7b
