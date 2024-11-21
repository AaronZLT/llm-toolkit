#!/bin/bash
set -x

### Global env setting
LLM_TOOLKIT_PATH=/hpc2hdd/home/lzhang330/llm-toolkit
MODELS_PATH=/hpc2hdd/home/lzhang330/asset/models

DATASET=$LLM_TOOLKIT_PATH/datasets
LLAMA1_3B=/hpc2hdd/home/lzhang330/llm-toolkit/models/Llama-2-1.3b-hf
LLAMA2_7B=$MODELS_PATH/Llama-2-7b-hf
LLAMA2_13B=$MODELS_PATH/Llama-2-13b-hf
LLAMA2_70B=$MODELS_PATH/Llama-2-70b-chat-hf
PHI3MINI4K=$MODELS_PATH/Phi-3-mini-4k-instruct
PHI3MINI128K=$MODELS_PATH/Phi-3-mini-128k-instruct

GPT2XL=$MODELS_PATH/gpt2-xl

F=" --flash_attn True"
R=" --gradient_checkpointing True"

L=" --peft lora --lora_r 128"
D=" --peft dora --lora_r 128"
LFA=" --peft lora-fa --lora_r 128"
V=" --peft vera --lora_r 128"
QL=" --quant True --double_quant True --bits 4 --peft lora --lora_r 64"
QL8=" --quant True --double_quant True --bits 8 --peft lora --lora_r 64"
QL8FA=" --quant True --double_quant True --bits 8 --peft lora-fa --lora_r 64"

PFX=" --peft prefix"
PMT=" --peft prompt"
EMD=" --peft embedding"

Z2=" --deepspeed zero2.json"
Z2O=" --deepspeed zero2off.json"
Z3=" --deepspeed zero3.json"
Z3O=" --deepspeed zero3off.json"

wandb offline


CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch finetune.py --dataset_name_or_path gsm8k --output_dir nf4-7e-6 --logging_strategy steps --logging_steps 1 --save_strategy epoch --save_steps 100 --dataloader_num_workers 1 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 2 --learning_rate 7e-6 --per_device_train_batch_size 1 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --quant True --bits 4 --peft lora --lora_r 128

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch finetune.py --dataset_name_or_path gsm8k --output_dir nf4-9e-6 --logging_strategy steps --logging_steps 1 --save_strategy epoch --save_steps 100 --dataloader_num_workers 1 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 2 --learning_rate 9e-6 --per_device_train_batch_size 1 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --quant True --bits 4 --peft lora --lora_r 128

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch finetune.py --dataset_name_or_path gsm8k --output_dir nf4-1e-5 --logging_strategy steps --logging_steps 1 --save_strategy epoch --save_steps 100 --dataloader_num_workers 1 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 2 --learning_rate 1e-5 --per_device_train_batch_size 1 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --quant True --bits 4 --peft lora --lora_r 128

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch finetune.py --dataset_name_or_path gsm8k --output_dir nf4-3e-5 --logging_strategy steps --logging_steps 1 --save_strategy epoch --save_steps 100 --dataloader_num_workers 1 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 2 --learning_rate 3e-5 --per_device_train_batch_size 1 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --quant True --bits 4 --peft lora --lora_r 128

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch finetune.py --dataset_name_or_path gsm8k --output_dir nf4-5e-5 --logging_strategy steps --logging_steps 1 --save_strategy epoch --save_steps 100 --dataloader_num_workers 1 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 2 --learning_rate 5e-5 --per_device_train_batch_size 1 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --quant True --bits 4 --peft lora --lora_r 128

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch finetune.py --dataset_name_or_path gsm8k --output_dir nf4-7e-5 --logging_strategy steps --logging_steps 1 --save_strategy epoch --save_steps 100 --dataloader_num_workers 1 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 2 --learning_rate 7e-5 --per_device_train_batch_size 1 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --quant True --bits 4 --peft lora --lora_r 128

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch finetune.py --dataset_name_or_path gsm8k --output_dir nf4-9e-5 --logging_strategy steps --logging_steps 1 --save_strategy epoch --save_steps 100 --dataloader_num_workers 1 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 2 --learning_rate 9e-5 --per_device_train_batch_size 1 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --quant True --bits 4 --peft lora --lora_r 128

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch finetune.py --dataset_name_or_path gsm8k --output_dir nf4-1e-4 --logging_strategy steps --logging_steps 1 --save_strategy epoch --save_steps 100 --dataloader_num_workers 1 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 2 --learning_rate 1e-4 --per_device_train_batch_size 1 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --quant True --bits 4 --peft lora --lora_r 128

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch finetune.py --dataset_name_or_path gsm8k --output_dir nf4-2e-4 --logging_strategy steps --logging_steps 1 --save_strategy epoch --save_steps 100 --dataloader_num_workers 1 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 2 --learning_rate 2e-4 --per_device_train_batch_size 1 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --quant True --bits 4 --peft lora --lora_r 128

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch finetune.py --dataset_name_or_path gsm8k --output_dir nf4-3e-4 --logging_strategy steps --logging_steps 1 --save_strategy epoch --save_steps 100 --dataloader_num_workers 1 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 2 --learning_rate 3e-4 --per_device_train_batch_size 1 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --quant True --bits 4 --peft lora --lora_r 128

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch finetune.py --dataset_name_or_path gsm8k --output_dir nf4-4e-4 --logging_strategy steps --logging_steps 1 --save_strategy epoch --save_steps 100 --dataloader_num_workers 1 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 2 --learning_rate 4e-4 --per_device_train_batch_size 1 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --quant True --bits 4 --peft lora --lora_r 128

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch finetune.py --dataset_name_or_path gsm8k --output_dir nf4-5e-4 --logging_strategy steps --logging_steps 1 --save_strategy epoch --save_steps 100 --dataloader_num_workers 1 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 2 --learning_rate 5e-4 --per_device_train_batch_size 1 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --quant True --bits 4 --peft lora --lora_r 128

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch finetune.py --dataset_name_or_path gsm8k --output_dir nf4-6e-4 --logging_strategy steps --logging_steps 1 --save_strategy epoch --save_steps 100 --dataloader_num_workers 1 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 2 --learning_rate 6e-4 --per_device_train_batch_size 1 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --quant True --bits 4 --peft lora --lora_r 128

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch finetune.py --dataset_name_or_path gsm8k --output_dir nf4-7e-4 --logging_strategy steps --logging_steps 1 --save_strategy epoch --save_steps 100 --dataloader_num_workers 1 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 2 --learning_rate 7e-4 --per_device_train_batch_size 1 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --quant True --bits 4 --peft lora --lora_r 128

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch finetune.py --dataset_name_or_path gsm8k --output_dir nf4-8e-4 --logging_strategy steps --logging_steps 1 --save_strategy epoch --save_steps 100 --dataloader_num_workers 1 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 2 --learning_rate 8e-4 --per_device_train_batch_size 1 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --quant True --bits 4 --peft lora --lora_r 128

CUDA_VISIBLE_DEVICES=0 DS_SKIP_CUDA_CHECK=1 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 accelerate launch finetune.py --dataset_name_or_path gsm8k --output_dir nf4-9e-4 --logging_strategy steps --logging_steps 1 --save_strategy epoch --save_steps 100 --dataloader_num_workers 1 --remove_unused_columns False --do_train --ddp_find_unused_parameters False --overwrite_output_dir --bf16 --profiler no --profiler_warmup_step 3 --max_steps -1 --hard_padding False --save_total_limit 3 --num_train_epochs 2 --learning_rate 9e-4 --per_device_train_batch_size 1 --source_max_len 512 --target_max_len 512 --model_name_or_path $LLAMA2_7B --flash_attn True --quant True --bits 4 --peft lora --lora_r 128

# --quant_storage bfloat16

cd ~/grabGPU && ./gg 75 168 0