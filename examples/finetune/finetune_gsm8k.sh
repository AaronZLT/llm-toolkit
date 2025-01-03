#!/bin/bash

LLAMA2_7B=meta-llama/Llama-2-7b-hf

wandb offline

# full finetune
CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 accelerate launch finetune.py --dataset_name_or_path metamath40k --output_dir fullft-metamath40k-output --logging_strategy steps --logging_steps 1 --save_strategy epoch --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 True --tf32 True --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 3 --learning_rate 3e-4 --per_device_train_batch_size 16 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --report_to wandb --gradient_checkpointing True

# lora finetune
CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 accelerate launch finetune.py --dataset_name_or_path metamath40k --output_dir lora-metamath40k-output --logging_strategy steps --logging_steps 1 --save_strategy epoch --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 True --tf32 True --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 3 --learning_rate 3e-4 --per_device_train_batch_size 16 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --report_to wandb --peft lora --lora_rank 16 --lora_scale 2.0

# qlora finetune
CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 accelerate launch finetune.py --dataset_name_or_path metamath40k --output_dir qlora-metamath40k-output --logging_strategy steps --logging_steps 1 --save_strategy epoch --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 True --tf32 True --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 3 --learning_rate 3e-4 --per_device_train_batch_size 16 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --report_to wandb --peft lora --lora_rank 16 --lora_scale 2.0 --quant True --bits 4