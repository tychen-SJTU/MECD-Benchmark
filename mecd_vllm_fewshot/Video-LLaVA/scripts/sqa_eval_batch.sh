#!/bin/bash

CHUNKS=8
for IDX in {0..7}; do
    CUDA_VISIBLE_DEVICES=$IDX python -m llava.eval.model_vqa_science \
        --model-path your_path/llava-lcs558k-scienceqa-vicuna-13b-v1.3 \
        --question-file ~/your_path/datasets/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json \
        --image-folder ~/your_path/datasets/ScienceQA/data/scienceqa/images/test \
        --answers-file ./test_llava-13b-chunk$CHUNKS_$IDX.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode llava_v1 &
done
