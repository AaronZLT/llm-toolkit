#!/bin/bash

LLM_BENCHMARK_PATH=YOUR_PATH


DATASET=$LLM_BENCHMARK_PATH/datasets
MODEL=$LLM_BENCHMARK_PATH/models/Llama-2-7b-hf

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch benchmark.py --data_path $DATASET --dataset alpaca-dummy --output_dir output --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --evaluation_strategy no --eval_steps 1 --eval_dataset_size 10 --max_eval_samples 10 --per_device_eval_batch_size 1 --max_new_tokens 32 --dataloader_num_workers 1 --remove_unused_columns False --do_train --do_eval --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler pytorch --profiler_warmup_step 3 --max_steps 5 --model_name_or_path $MODEL --per_device_train_batch_size 1 --source_max_len 256 --target_max_len 256 --max_memory_MB 80000
