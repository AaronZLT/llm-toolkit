#!/bin/bash

#
# This is an example, for sparse lora fine-tuning Llama-2-7b-chat-hf, on mmlu
# For more datasets, please refer to the examples in examples/finetune
# Remember to change max_steps to -1 for fine-tuning
#

wandb offline

LLAMA2_7B_CHAT=/mnt/sdb/zhanglongteng/sdd/zhanglongteng/Llama-2-7b-chat-hf

#
# A recommanded sparse setting is as follows:
# sparsity_ratio: [0.1, 0.2, 0.3] -> sparse_warmup_ratio: 0.1 & sparse_warmup_steps: 1
# sparsity_ratio: [0.4, 0.5, 0.6] -> sparse_warmup_ratio: 0.2 & sparse_warmup_steps: 2
# sparsity_ratio: [0.7, 0.8, 0.9] -> sparse_warmup_ratio: 0.3 & sparse_warmup_steps: 3
# You can also try if one-shot sparse at the begining is better, i.e., sparse_warmup_ratio: 0 & sparse_warmup_steps: 1
#

#
# lora with sparse fine-tuning
#
CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 accelerate launch finetune.py --dataset_name_or_path mmlu --output_dir llama2_7b.mmlu.lora.sparse.output --logging_strategy steps --logging_steps 1 --save_strategy epoch --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 True --tf32 True --max_steps 5 --hard_padding False --save_total_limit 3 --num_train_epochs 3 --learning_rate 7e-5 --per_device_train_batch_size 16 --source_max_len 896 --target_max_len 128 --model_name_or_path $LLAMA2_7B --flash_attn True --report_to wandb --gradient_checkpointing True --peft lora --lora_rank 16 --lora_scale 2.0 --sparse True --sparsity_ratio 0.5 --sparse_warmup_ratio 0.2 --sparse_warmup_steps 1


#
# lora with sparse quantization fine-tuning
# W = Quant(sparse(W)) + LoRA
#
CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 accelerate launch finetune.py --dataset_name_or_path mmlu --output_dir llama2_7b.mmlu.lora.sparse.output --logging_strategy steps --logging_steps 1 --save_strategy epoch --dataloader_num_workers 32 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 True --tf32 True --max_steps 5 --hard_padding False --save_total_limit 3 --num_train_epochs 3 --learning_rate 7e-5 --per_device_train_batch_size 16 --source_max_len 896 --target_max_len 128 --model_name_or_path $LLAMA2_7B --flash_attn True --report_to wandb --gradient_checkpointing True --peft lora --lora_rank 16 --lora_scale 2.0 --sparse True --sparsity_ratio 0.5 --sparse_warmup_ratio 0.2 --sparse_warmup_steps 1 --quant bnb --bits 4
