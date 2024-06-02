#!/bin/bash

# HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LLM_BENCHMARK_PATH=/hpc2hdd/home/lzhang330/llm-toolkit

DATASET=$LLM_BENCHMARK_PATH/datasets
MODEL=/hpc2hdd/home/lzhang330/ssd_workspace/models/Llama-2-7b-hf

# gsm8k lora
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DS_SKIP_CUDA_CHECK=1 accelerate launch finetune.py --dataset gsm8k --output_dir gsm8k_3e-4_lora --logging_strategy steps --logging_steps 1 --save_strategy steps --save_steps 10 --evaluation_strategy no --max_new_tokens 32 --dataloader_num_workers 1 --remove_unused_columns False --do_train --do_eval --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --model_name_or_path $MODEL --per_device_train_batch_size 128 --source_max_len 256 --target_max_len 256 --max_memory_MB 80000 --use_lora True --fa False --lora_r 64 --lora_alpha 128 --learning_rate 3e-4 --gradient_checkpointing True --max_steps -1 --num_train_epochs 50

# hellaswag lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DS_SKIP_CUDA_CHECK=1 accelerate launch finetune.py --dataset hellaswag --output_dir hellaswag_3e-4_lora --logging_strategy steps --logging_steps 1 --save_strategy steps --save_steps 20 --evaluation_strategy no --max_new_tokens 32 --dataloader_num_workers 1 --remove_unused_columns False --do_train --do_eval --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --model_name_or_path $MODEL --per_device_train_batch_size 128 --source_max_len 256 --target_max_len 256 --max_memory_MB 80000 --use_lora True --fa False --lora_r 64 --lora_alpha 128 --learning_rate 3e-4 --gradient_checkpointing True --max_steps -1 --num_train_epochs 2

# --max_steps 5