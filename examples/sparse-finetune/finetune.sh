#!/bin/bash

LLAMA2_7B=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Llama-2-7b-hf

wandb offline


# This is an example, for fine-tuning meta-llama/Meta-Llama-3-8B-Instruct, with base model quantized to NF4, and lora rank 16, dynamic sparse with 0.5 sparsity ratio, and warmup ratio 0.2, and warmup steps 1
# i.e., W = Quant(sparse(W)) + LoRA
CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 accelerate launch finetune.py --dataset_name_or_path mmlu --output_dir validate --overwrite_output_dir --do_train --logging_strategy steps --logging_steps 1 --save_strategy epoch --save_total_limit 3 --max_steps 100 --num_train_epochs 3 --dataloader_num_workers 32 --remove_unused_columns False --ddp_find_unused_parameters False --bf16 True --tf32 True  --learning_rate 7e-5 --per_device_train_batch_size 1 --source_max_len 512 --target_max_len 512 --hard_padding False --model_name_or_path /mnt/sdb/zhanglongteng/sdd/zhanglongteng/llama-3-8B-Instruct --flash_attn True --gradient_checkpointing True --report_to wandb --peft lora --lora_rank 16 --lora_scale 2.0 --quant True --bits 4 --sparse True --sparse_type dynamic_sparse --sparsity_ratio 0.5 --sparse_warmup_ratio 0.2 --sparse_warmup_steps 1