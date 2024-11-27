#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
DS_SKIP_CUDA_CHECK=1 \
HF_DATASETS_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
accelerate launch benchmark.py \
    --dataset alpaca \
    --output_dir profiler \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy no \
    --dataloader_num_workers 32 \
    --remove_unused_columns False \
    --do_train \
    --ddp_find_unused_parameters False \
    --overwrite_output_dir \
    --bf16 \
    --profiler pytorch \
    --profiler_warmup_step 3 \
    --max_steps 5 \
    --hard_padding True \
    --model_name_or_path $LLAMA2_1_3B \
    --source_max_len 512 --target_max_len 512 \
    --per_device_train_batch_size 16
