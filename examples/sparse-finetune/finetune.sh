#!/bin/bash

LLAMA2_7B=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Llama-2-7b-hf

wandb offline


CUDA_VISIBLE_DEVICES=7 DS_SKIP_CUDA_CHECK=1 accelerate launch finetune.py --dataset_name_or_path metamath40k --output_dir finetune.llama2_7b.metamath40k.4bit.dynamic_sparse.sparse_warmup_steps1 --logging_strategy steps --logging_steps 1 --save_strategy epoch --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 True --tf32 True --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 3 --learning_rate 3e-4 --per_device_train_batch_size 1 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --gradient_checkpointing True --report_to wandb --peft lora --lora_rank 64 --lora_scale 2.0 --quant True --bits 4 --sparse True --sparse_type dynamic_sparse --sparsity_ratio 0.5 --sparse_warmup_ratio 0.4 --sparse_warmup_steps 1

