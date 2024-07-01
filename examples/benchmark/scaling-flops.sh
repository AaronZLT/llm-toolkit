#!/bin/bash

DATASET=/hpc2hdd/home/lzhang330/llm-benchmark/datasets

LLAMA7B=/hpc2hdd/home/lzhang330/ssd_workspace/models/Llama-2-7b-hf
LLAMA13B=/hpc2hdd/home/lzhang330/ssd_workspace/models/Llama-2-13b-hf
LLAMA70B=/hpc2hdd/home/lzhang330/ssd_workspace/models/Llama-2-70b-chat-hf

techniques=(
    ""
    " --flash_attn True"
    " --gradient_checkpointing True"
    " --use_lora True --lora_r 64"
    " --quant True --double_quant True --bits 4 --use_lora True --lora_r 64"
)

    # " --deepspeed zero2.json"
    # " --deepspeed zero2off.json"
    # " --deepspeed zero3.json"
    # " --deepspeed zero3off.json"
QUANT=" --quant True --double_quant True --bits 4"

model_sizes=(
    " --model_name_or_path \$LLAMA7B"
    " --model_name_or_path \$LLAMA13B"
    " --model_name_or_path \$LLAMA70B"
)

sequence_lengths=(
    " --source_max_len 128 --target_max_len 128"
    " --source_max_len 256 --target_max_len 256"
    " --source_max_len 512 --target_max_len 512"
    " --source_max_len 1024 --target_max_len 1024"
    " --source_max_len 2048 --target_max_len 2048"
)

base="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch benchmark.py --data_path $DATASET --dataset alpaca --output_dir scaling-flops --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps 5 --per_device_train_batch_size 128 --max_memory_MB 80000 --hard_padding True --auto_find_batch_size True"

#256
CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch benchmark.py --data_path $DATASET --dataset alpaca --output_dir scaling-flops --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps 5 --per_device_train_batch_size 1 --max_memory_MB 80000 --hard_padding True --source_max_len 128 --target_max_len 128 --model_name_or_path $LLAMA7B

#512
CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch benchmark.py --data_path $DATASET --dataset alpaca --output_dir scaling-flops --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps 5 --per_device_train_batch_size 2 --max_memory_MB 80000 --hard_padding True --source_max_len 128 --target_max_len 128 --model_name_or_path $LLAMA7B

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch benchmark.py --data_path $DATASET --dataset alpaca --output_dir scaling-flops --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps 5 --per_device_train_batch_size 1 --max_memory_MB 80000 --hard_padding True --source_max_len 256 --target_max_len 256 --model_name_or_path $LLAMA7B

#1024
CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch benchmark.py --data_path $DATASET --dataset alpaca --output_dir scaling-flops --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps 5 --per_device_train_batch_size 4 --max_memory_MB 80000 --hard_padding True --source_max_len 128 --target_max_len 128 --model_name_or_path $LLAMA7B

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch benchmark.py --data_path $DATASET --dataset alpaca --output_dir scaling-flops --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps 5 --per_device_train_batch_size 2 --max_memory_MB 80000 --hard_padding True --source_max_len 256 --target_max_len 256 --model_name_or_path $LLAMA7B

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch benchmark.py --data_path $DATASET --dataset alpaca --output_dir scaling-flops --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps 5 --per_device_train_batch_size 1 --max_memory_MB 80000 --hard_padding True --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA7B

#2048
CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch benchmark.py --data_path $DATASET --dataset alpaca --output_dir scaling-flops --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps 5 --per_device_train_batch_size 8 --max_memory_MB 80000 --hard_padding True --source_max_len 128 --target_max_len 128 --model_name_or_path $LLAMA7B

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch benchmark.py --data_path $DATASET --dataset alpaca --output_dir scaling-flops --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps 5 --per_device_train_batch_size 4 --max_memory_MB 80000 --hard_padding True --source_max_len 256 --target_max_len 256 --model_name_or_path $LLAMA7B

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch benchmark.py --data_path $DATASET --dataset alpaca --output_dir scaling-flops --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps 5 --per_device_train_batch_size 2 --max_memory_MB 80000 --hard_padding True --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA7B

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch benchmark.py --data_path $DATASET --dataset alpaca --output_dir scaling-flops --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps 5 --per_device_train_batch_size 1 --max_memory_MB 80000 --hard_padding True --source_max_len 1024 --target_max_len 1024 --model_name_or_path $LLAMA7B

#4096
CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch benchmark.py --data_path $DATASET --dataset alpaca --output_dir scaling-flops --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps 5 --per_device_train_batch_size 16 --max_memory_MB 80000 --hard_padding True --source_max_len 128 --target_max_len 128 --model_name_or_path $LLAMA7B

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch benchmark.py --data_path $DATASET --dataset alpaca --output_dir scaling-flops --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps 5 --per_device_train_batch_size 8 --max_memory_MB 80000 --hard_padding True --source_max_len 256 --target_max_len 256 --model_name_or_path $LLAMA7B

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch benchmark.py --data_path $DATASET --dataset alpaca --output_dir scaling-flops --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps 5 --per_device_train_batch_size 4 --max_memory_MB 80000 --hard_padding True --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA7B

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch benchmark.py --data_path $DATASET --dataset alpaca --output_dir scaling-flops --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps 5 --per_device_train_batch_size 2 --max_memory_MB 80000 --hard_padding True --source_max_len 1024 --target_max_len 1024 --model_name_or_path $LLAMA7B

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 accelerate launch benchmark.py --data_path $DATASET --dataset alpaca --output_dir scaling-flops --logging_strategy steps --logging_steps 1 --save_strategy no --save_steps 1 --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps 5 --per_device_train_batch_size 1 --max_memory_MB 80000 --hard_padding True --source_max_len 2048 --target_max_len 2048 --model_name_or_path $LLAMA7B